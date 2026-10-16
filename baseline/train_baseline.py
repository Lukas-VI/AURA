"""AURA 基线训练脚本。

论文出处：Liu et al. 2026, 3.3 / 3.4。按论文修正后的训练设置：
    阶段 1（无监督，式 11）：L1 = λr·Lr + λh·Lh。
        Lr = 前景重构 L2 + 背景重构 L2（模型内部分别重构前后景）；
        Lh = 最小熵比特损失（作用于前后景拼接后的哈希特征）。
    阶段 2（弱监督，式 16）：L2 = λr·Lr + λh·Lh + λc·Lc + λq·Lquadruplet。
        Lc = 检测块对真实/伪造的 BCE（检测块只读前景哈希特征）；
        Lquadruplet = 前景项(前景哈希码) + 背景项(背景哈希码)——两项分开计算。
    （额外的 λb·比特平衡项为本项目为防止哈希坍缩所加的工程项，论文没有。）

数据：使用 FF++ 抽帧结果（D:/ffpp_frames，见 data/dataset.py）。
      伪造样本直接用【真实换脸算法生成的伪造帧】（q+2），检测监督因此能看到真实伪影；
      只有前景正样本 q+1（保留 q 的人脸、换到供体背景）仍用合成换脸在线构造。

用法（在项目根目录 d:\py\AURA 下运行）：
    python -m baseline.train_baseline --config configs/baseline.yaml
    python -m baseline.train_baseline --config configs/baseline.yaml --epochs_stage1 2 --epochs_stage2 2 --limit 64
"""
import argparse
import json
import os

# 必须在 import torch 之前设置：4GB 显存下开启可扩展显存段，缓解碎片导致的 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from baseline.data import QuadrupletDataset, SyntheticFaceSwap, build_datasets
from baseline.losses import bit_balance_loss, min_entropy_loss, quadruplet_loss, reconstruction_loss
from baseline.models import HashVAE
from baseline.utils import ensure_dir, get_device, load_config, resolve_path, set_seed

# ---------------- 张量 <-> OpenCV 批量互转（用于在训练中在线换脸） ----------------
# 关键优化：整批一次性搬到 CPU，而不是逐样本搬运。逐样本 .cpu() 会强制 GPU 同步，
# 在 stage2 里每步要处理多批张量，逐样本同步会让速度下降一个数量级。

def _batch_to_bgr(t):
    """(B,3,H,W) RGB float[0,1] -> (B,H,W,3) BGR uint8 的 numpy（一次 CPU 拷贝）。"""
    arr = (t.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return np.stack([cv2.cvtColor(a, cv2.COLOR_RGB2BGR) for a in arr], axis=0)


def _batch_from_bgr(arr, device):
    """(B,H,W,3) BGR uint8 -> (B,3,H,W) RGB float[0,1] 张量（一次 GPU 拷贝）。"""
    rgb = np.stack([cv2.cvtColor(a, cv2.COLOR_BGR2RGB) for a in arr], axis=0)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    return t.to(device)


def tensor_swap(faker: SyntheticFaceSwap, src: torch.Tensor, donor: torch.Tensor,
                src_mask: torch.Tensor = None) -> torch.Tensor:
    """在一个 batch 上执行合成换脸，输入输出均为 (B,3,H,W) 的 RGB 张量。

    src_mask 为该批图的前景掩膜 (B,1,H,W)；传入可复用数据集已算好的解析掩膜，
    避免训练过程中重复跑人脸解析网络。
    """
    sb = _batch_to_bgr(src)            # 整批搬到 CPU
    db = _batch_to_bgr(donor)
    sm = src_mask.detach().cpu().numpy() if src_mask is not None else None
    outs = np.empty_like(sb)           # 复用一块内存，避免反复分配
    for i in range(sb.shape[0]):
        outs[i] = faker.swap(sb[i], db[i], None if sm is None else sm[i, 0])
    return _batch_from_bgr(outs, src.device)


# ---------------- 训练流程 ----------------

def train_stage1(model, loader, optimizer, cfg, device, epoch, writer=None):
    """阶段 1：前景重构 + 背景重构 + 最小熵 + 比特平衡（论文式 11）。"""
    model.train()
    total_r, total_h, total_b, n = 0.0, 0.0, 0.0, 0
    hash_batches = []                                  # 收集哈希特征，用于统计是否坍缩
    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)                            # 逐样本前景掩膜（BiSeNet 解析）
        out = model(img, mask=mask, need_recon=True, need_pred=False)   # 阶段1无需检测分支

        # 重构损失 = 前景 L2 + 背景 L2（论文式 9）；前后景目标由传入的前景掩膜给出
        loss_r = (reconstruction_loss(out["recon_f"], out["fg"]) +
                  reconstruction_loss(out["recon_b"], out["bg"]))
        loss_h = min_entropy_loss(out["hash"])         # 作用于前后景拼接哈希
        loss_b = bit_balance_loss(out["hash"])         # 防止所有样本被编码成同一个码
        loss = (cfg["lambda_recon"] * loss_r + cfg["lambda_entropy"] * loss_h +
                cfg.get("lambda_balance", 0.1) * loss_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_r += loss_r.item() * img.shape[0]
        total_h += loss_h.item() * img.shape[0]
        total_b += loss_b.item() * img.shape[0]
        n += img.shape[0]
        hash_batches.append(out["hash"].detach().cpu())

    avg_r, avg_h, avg_b = total_r / max(n, 1), total_h / max(n, 1), total_b / max(n, 1)
    # 哈希特征统计：min/max 逼近 0/1 说明已二值化；bit_mean_dev 越小说明各比特越均衡、越不坍缩
    h = torch.cat(hash_batches, dim=0)
    dev = (h.mean(dim=0) - 0.5).abs().mean().item()
    print(f"  [Stage1] epoch {epoch}: loss_r={avg_r:.4f}, loss_h={avg_h:.4f}, loss_b={avg_b:.4f} "
          f"| hash[min={h.min():.3f}, max={h.max():.3f}, mean={h.mean():.3f}, bit_dev={dev:.3f}]")
    if writer:
        writer.add_scalar("stage1/recon", avg_r, epoch)
        writer.add_scalar("stage1/entropy", avg_h, epoch)
        writer.add_scalar("stage1/balance", avg_b, epoch)
    return {"recon": avg_r, "entropy": avg_h, "balance": avg_b}


def train_stage2(model, loader, optimizer, faker, cfg, device, epoch, writer=None):
    """阶段 2：重构 + 最小熵 + 比特平衡 + 四元组 + 检测（论文式 15/16）。

    四元组样本（论文 3.1 的相似度定义）：
        q     = 伪造图的【背景来源真实图】（img_q）
        q+1   = 保留 q 的人脸、换到供体背景 —— 由合成换脸实时构造（前景正样本）；
        q+2   = 【真实伪造图】（img_fake），其背景来自 q、人脸来自供体视频 Y（背景正样本）；
        q−    = 供体视频 Y 的真实图（img_donor），背景与身份都不同（负样本）。
    检测监督：q 记为真(label=0)，真实伪造图 q+2 记为假(label=1)。因为 q+2 是 FF++ 用真实
    换脸算法生成的，前景里带有真实的融合伪影，DDB 才有可学的信号。
    """
    model.train()
    acc = {k: 0.0 for k in ["loss", "recon", "entropy", "balance", "quad", "cls"]}
    n = 0
    m_fg = float(cfg.get("margin_fg", 0.5))     # 论文 α
    m_bg = float(cfg.get("margin_bg", 0.5))     # 论文 β
    for batch in loader:
        img_q = batch["img_q"].to(device)      # q：真实源图（伪造图的背景来源）
        img_f = batch["img_fake"].to(device)   # q+2：真实伪造图
        img_d = batch["img_donor"].to(device)  # q−：供体视频的真实图
        mask_q = batch["mask_q"].to(device)    # 三张图各自的前景掩膜
        mask_f = batch["mask_fake"].to(device)
        mask_d = batch["mask_donor"].to(device)

        # q+1：保留 q 的人脸身份、换到供体背景。swap(源=供体图, 供体=img_q) -> 背景来自供体、人脸来自 q
        # 合成区域沿用【供体图 img_d 的前景掩膜】：贴脸后前景位置不变，掩膜照旧适用
        q1 = tensor_swap(faker, img_d, img_q, mask_d)

        # 【关键】q 与真实伪造图必须拼成一个【混合批】一次编码。
        # 原因：hash_proj 末端的 BatchNorm1d 在训练态按“批内统计量”归一化。若真/假各自单独成批
        # （类纯 + batch=2），每维会被归一成恰好 ±1，哈希退化成只反映“批内两张图的相对次序”，
        # 不再是单张图自身的函数——DDB 因此拿不到任何真假信号（实测 cls 恒为 ln2≈1.386、
        # quad 在 0.44↔1.64 剧烈震荡、换脸后硬量化码变化 0 位）。混合成一批后 BN 同时看到两类，
        # 哈希重新成为单张图的可微函数，检测才有可回传的梯度。
        enc_qf = model.encode(torch.cat([img_q, img_f], dim=0),
                              mask=torch.cat([mask_q, mask_f], dim=0))
        h_f_q, h_f_f = enc_qf["h_f"].chunk(2, dim=0)     # 锚点前景哈希 / 真实伪造图前景哈希
        h_b_q, h_b_f = enc_qf["h_b"].chunk(2, dim=0)     # 锚点背景哈希 / 真实伪造图背景哈希
        fg_q, _ = enc_qf["fg"].chunk(2, dim=0)           # 重构目标只取 q 这一半
        bg_q, _ = enc_qf["bg"].chunk(2, dim=0)

        # q+1 / q− 只需哈希值（作为四元组的另一端），不保留计算图以省显存
        # （4GB 卡上 batch 4 保留两张图会溢出到共享内存，单步从 0.9s 掉到 7s）。
        with torch.no_grad():
            enc_pn = model.encode(torch.cat([q1, img_d], dim=0),
                                  mask=torch.cat([mask_d, mask_d], dim=0))
            h_f_p1, h_f_n = enc_pn["h_f"].chunk(2, dim=0)
            h_b_p1, h_b_n = enc_pn["h_b"].chunk(2, dim=0)

        # 重构/检测按需在各半段上单独跑，避免对不需要的样本浪费显存
        hash_q = torch.cat([h_f_q, h_b_q], dim=1)                 # 拼接哈希，供最小熵/比特平衡
        recon_f = model.decoder(h_f_q)                            # ~x_f(q)
        recon_b = model.decoder(h_b_q)                            # ~x_b(q)
        pred_q = model.detector(h_f_q)                            # q 的伪造概率（标签 0）
        pred_f = model.detector(h_f_f)                            # q+2 的伪造概率（标签 1）

        loss_r = (reconstruction_loss(recon_f, fg_q) +
                  reconstruction_loss(recon_b, bg_q))
        loss_h = min_entropy_loss(hash_q)
        loss_b = bit_balance_loss(hash_q)
        # 四元组（论文式 15）：前景项对 q 与 q+1 的【前景哈希】，背景项对 q 与 q+2 的【背景哈希】
        loss_quad = quadruplet_loss(h_f_q, h_f_p1, h_b_q, h_b_f,
                                    h_f_n.detach(), h_b_n.detach(),
                                    margin_fg=m_fg, margin_bg=m_bg)
        # 检测：真实图 q 标签 0，真实伪造图 q+2 标签 1（DDB 只读前景哈希）
        label_real = torch.zeros_like(pred_q)
        label_fake = torch.ones_like(pred_f)
        loss_c = (torch.nn.functional.binary_cross_entropy(pred_q, label_real) +
                  torch.nn.functional.binary_cross_entropy(pred_f, label_fake))

        loss = (cfg["lambda_recon"] * loss_r + cfg["lambda_entropy"] * loss_h +
                cfg.get("lambda_balance", 0.1) * loss_b +
                cfg["lambda_quadruplet"] * loss_quad + cfg["lambda_detection"] * loss_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = img_q.shape[0]
        acc["loss"] += loss.item() * bs
        acc["recon"] += loss_r.item() * bs
        acc["entropy"] += loss_h.item() * bs
        acc["balance"] += loss_b.item() * bs
        acc["quad"] += loss_quad.item() * bs
        acc["cls"] += loss_c.item() * bs
        n += bs

    for k in acc:
        acc[k] /= max(n, 1)
    print(f"  [Stage2] epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in acc.items()))
    if writer:
        for k, v in acc.items():
            writer.add_scalar(f"stage2/{k}", v, epoch)
    return acc


def main():
    parser = argparse.ArgumentParser(description="AURA 基线训练")
    parser.add_argument("--config", default="configs/baseline.yaml", help="配置文件路径")
    parser.add_argument("--epochs_stage1", type=int, default=None, help="覆盖阶段1轮数")
    parser.add_argument("--epochs_stage2", type=int, default=None, help="覆盖阶段2轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--limit", type=int, default=None, help="每个身份只用前 N 张做快速冒烟测试")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="临时覆盖任意配置项，可重复。例如 --set lambda_entropy=0")
    parser.add_argument("--init_from", default=None,
                        help="从已有权重热启动（相对 baseline 包目录或绝对路径，如 checkpoints/baseline.pth）")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="跳过阶段1，直接用 --init_from 的权重做阶段2（消融实验省时间）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs_stage1 is not None:
        cfg["epochs_stage1"] = args.epochs_stage1
    if args.epochs_stage2 is not None:
        cfg["epochs_stage2"] = args.epochs_stage2
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size            # 显式指定时两个阶段都用同一个值
        cfg["stage2_batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.limit is not None:
        cfg["num_train_videos"] = args.limit         # 冒烟测试：缩小训练视频数
        cfg["num_val_videos"] = max(args.limit // 4, 1)
        cfg["num_test_videos"] = max(args.limit // 4, 1)
    for kv in args.set:                              # 逐一覆盖任意配置项（消融实验用）
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
        print(f"[override] {k} = {cfg[k]}")

    set_seed(int(cfg.get("seed", 42)))            # 固定随机种子，保证可复现
    device = get_device(cfg)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"device = {device} ({gpu_name}) | torch = {torch.__version__}")   # 记录实际使用的卡

    # ---------- 数据 ----------
    datasets, index = build_datasets(cfg)
    train_ds = datasets["train"]
    # 先把训练集（含源图/供体图）的解析掩膜缓存建好，再释放解析网络——避免它与 HashVAE 争 4GB 显存
    if train_ds.warm_mask_cache():
        print(f"前景掩膜来源: {cfg.get('fg_mask_source', 'parse')}（缓存已就绪）")
    # 阶段1/阶段2 用不同的 batch：阶段2 同时保留锚点与伪造图两张计算图，显存占用更高
    bs1 = int(cfg["batch_size"])
    bs2 = int(cfg.get("stage2_batch_size", bs1))
    train_loader = DataLoader(train_ds, batch_size=bs1, shuffle=True,
                              num_workers=int(cfg.get("num_workers", 0)))
    # 阶段2 用的四元组数据集：只遍历真实伪造样本，附带源图/供体图
    quad_ds = QuadrupletDataset(train_ds)
    quad_loader = DataLoader(quad_ds, batch_size=bs2, shuffle=True,
                             num_workers=int(cfg.get("num_workers", 0)))
    print(f"训练集样本: {len(train_ds)} 张（含真实+伪造）| 四元组样本: {len(quad_ds)} "
          f"| batch: stage1={bs1}, stage2={bs2}")

    # ---------- 模型 / 优化器 ----------
    model = HashVAE.build_from_config(cfg).to(device)
    if args.init_from:                              # 热启动：复用已有阶段的权重（纯编排，不改结构）
        init_path = resolve_path(args.init_from)
        model.load_state_dict(torch.load(init_path, map_location=device, weights_only=True))
        print(f"已从权重热启动: {init_path}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 betas=tuple(cfg.get("betas", [0.9, 0.99])),
                                 weight_decay=cfg.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(cfg.get("step_size", 5)), gamma=float(cfg.get("gamma", 0.5)))

    # 在线换脸器：直接用数据集里的同一个（掩膜提供器与模型前景定义一致）
    faker = train_ds.faker

    # tensorboard 日志
    log_dir = ensure_dir(cfg.get("log_dir", "logs/"))
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    except Exception as e:  # tensorboard 不可用时降级为纯打印
        print(f"[warn] tensorboard 不可用，仅打印日志: {e}")

    history = {"stage1": [], "stage2": []}

    print("=== Stage 1: 无监督哈希学习 ===")
    if args.skip_stage1:
        print("  [skip] 已指定 --skip_stage1，跳过阶段1（请配合 --init_from 使用）")
    else:
        for epoch in range(1, int(cfg["epochs_stage1"]) + 1):
            history["stage1"].append(train_stage1(model, train_loader, optimizer, cfg, device, epoch, writer))
            scheduler.step()

    print("=== Stage 2: 弱监督（四元组 + 检测，伪造图来自 FF++） ===")
    for epoch in range(1, int(cfg["epochs_stage2"]) + 1):
        history["stage2"].append(train_stage2(model, quad_loader, optimizer, faker, cfg, device, epoch, writer))
        scheduler.step()

    # ---------- 保存 ----------
    save_dir = ensure_dir(cfg.get("save_dir", "checkpoints/"))
    name = cfg.get("name", "baseline")
    ckpt_path = os.path.join(save_dir, f"{name}.pth")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(save_dir, f"{name}_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"模型已保存: {ckpt_path}")
    if device.type == "cuda":
        print(f"GPU 峰值显存: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
