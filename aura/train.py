"""AURA 自建单编码器 · 全图对比学习训练入口。

方案（2026-10-16 拍板）：不跑 baseline 复现，直接自建单编码器；整图喂入、单阶段端到端、
监督对比学习 —— 让"同源不同攻击 / 退化的图"靠得近、"不同源"离得远。
主干保留论文全量规格（256px / M=5 / D=5 / 64ch / aff512 / embed2048）。

评测协议（与 baseline 完全一致）
--------------------------------
图库 = 测试集 200 个源视频 × 8 帧真实帧（1600），查询 = 测试集伪造帧（8000）；
命中 = 检索到的 Top-K 里含该查询的 source_id（帧级）或同源视频（视频级）。

kill 判据（写代码前已登记，跑 1–2 epoch 后裁决）
-----------------------------------------------
帧级 Top-1 必须高于：
  - 49.34%：零训练 64 位 pHash 的【背景】基线（原登记值）
  - 58.80%：零训练 32×32 缩略图【整图】基线（本轮"全图方法"的严格对照）
达不到就永久停掉这条线，回填 docs/AURA_HANDOFF.md。

用法（项目根目录 d:\py\AURA 下）
--------------------------------
    E:\conda\envs\pytorch\python.exe -m aura.train --config aura/configs/aura_full.yaml
    冒烟：--limit_videos 8 --steps_per_epoch 5 --epochs 1
"""
import argparse
import json
import math
import os
import time

# 必须在 import torch 之前设置：缓解显存碎片导致的 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aura.data import ContrastiveDataset, GeoPairDataset, GroupBatchSampler
from aura.encoder import SourceEncoder
from aura.eval import evaluate, evaluate_degradation, format_degradation, format_report
from aura.geo import geo_consistency_loss, grid_coords, pair_score
from aura.losses import grid_supcon_loss, supcon_loss, variance_loss
from baseline.data.dataset import build_ffpp_index
from baseline.utils import ensure_dir, get_device, load_config, set_seed

# kill 判据已于 2026-10-18 上移。手工描述子扫描（aura/runs/_probe_protocol_ceiling.py 的
# classical 段，读数落 runs/probe_protocol_ceiling.json）把零训练上限从 tiny32 的 58.71% 推到
# r8_norm（8×8 逐图标准化缩略图，仅 192 维）的 60.05%、以及 r32_norm+4*huehsv 融合的 60.62%。
# 旧的两条线（pHash64 背景 49.34% / tiny32 整图 58.80%）已被下面两条取代。
KILL_LINES = {"r8_norm_单描述子": 0.6005, "r32_norm+4*huehsv_融合": 0.6062}


def lowres_teacher(img, res):
    """蒸馏教师：逐图全局标准化的 res×res 缩略图（L2 归一化）。

    与零训练扫描脚本同配方（`_probe_protocol_ceiling.py` 里的 `r8_norm`）：
    area 降采样 → 逐图减均值除标准差 → L2 归一化。res=8 时是 192 维、
    零训练帧级 Top-1 = 60.05%，本协议下最强的【单条确定性】描述子。
    注意：标准化的维度顺序与扫描脚本不同（这里按通道优先，脚本按像素优先），
    但两者是同一个向量的置换，逐图标准化 + L2 归一化对置换等变，取值分布一致。
    """
    t = F.interpolate(img, size=(res, res), mode="area").reshape(img.shape[0], -1)
    t = (t - t.mean(dim=1, keepdim=True)) / (t.std(dim=1, keepdim=True) + 1e-6)
    return F.normalize(t, dim=1)


def _build_loaders(index, cfg, seed):
    """训练集 = 分组批采样；评估用的图库/查询 loader 为确定性（不加退化）。"""
    size = int(cfg["image_size"])
    train_ds = ContrastiveDataset(index["train"], image_size=size, train=True, seed=seed,
                                  erase_prob=float(cfg.get("erase_prob", 0.0)),
                                  mask_cache_dir=str(cfg.get("mask_cache_dir", "cache/masks")),
                                  label_key=str(cfg.get("label_key", "identity")),
                                  geom_aug=bool(cfg.get("geom_aug", False)))
    sampler = GroupBatchSampler(train_ds.labels,
                                groups_per_batch=int(cfg["groups_per_batch"]),
                                samples_per_group=int(cfg["samples_per_group"]),
                                steps_per_epoch=int(cfg["steps_per_epoch"]),
                                seed=seed,
                                video_keys=[s["source_id"] for s in train_ds.samples],
                                hard_neg_pair=bool(cfg.get("hard_neg_pair", False)))
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=int(cfg.get("num_workers", 0)))

    val_ds = ContrastiveDataset(index["val"], image_size=size, train=False, seed=seed)
    test_ds = ContrastiveDataset(index["test"], image_size=size, train=False, seed=seed)
    real = [s for s in index["test"] if s["kind"] == "real"]
    fake = [s for s in index["test"] if s["kind"] == "fake"]
    # 评估批大小必须小：encode_loader 里 model.eval() 会关掉梯度检查点（_block/_fusion_tail
    # 只在 training 时启用），256px 下 batch16 的无检查点前向会直接爆显存，故默认降到 4。
    eval_bs = int(cfg.get("eval_batch_size", 4))
    db_loader = DataLoader(ContrastiveDataset(real, image_size=size, train=False, seed=seed),
                           batch_size=eval_bs, shuffle=False, num_workers=0)
    query_loader = DataLoader(ContrastiveDataset(fake, image_size=size, train=False, seed=seed),
                              batch_size=eval_bs, shuffle=False, num_workers=0)

    # K 轮：可微几何验证模块的训练对（退化伪造查询 ↔ 同源干净真帧）。与检索批分开采样：
    # 检索批保持 I 轮原样（不变量），几何支路只增不改，便于把 T1 变化归因到几何监督。
    geo_loader = None
    if bool(cfg.get("geo", False)):
        geo_size = int(cfg.get("geo_size", size))
        geo_ds = GeoPairDataset([s for s in index["train"] if s["kind"] == "fake"],
                                image_size=geo_size, seed=seed,
                                max_deg=float(cfg.get("geo_max_deg", 12.0)),
                                smin=float(cfg.get("geo_smin", 0.7)), smax=float(cfg.get("geo_smax", 1.3)),
                                max_shift=float(cfg.get("geo_max_shift", 0.08)),
                                cmin=float(cfg.get("geo_cmin", 0.7)))
        geo_loader = DataLoader(geo_ds, batch_size=int(cfg.get("geo_batch", 4)),
                                shuffle=True, num_workers=0, drop_last=True)
    return train_ds, sampler, train_loader, val_ds, db_loader, query_loader, geo_loader


def train_one_epoch(model, sampler, train_loader, optimizer, scheduler, cfg, device,
                    epoch, queue_z, queue_labels, proj=None, writer=None, geo_loader=None):
    """单阶段对比学习：SupCon（+跨批内存） + 方差铰链（+可选低层蒸馏/可微几何一致性）。"""
    model.train()
    if bool(cfg.get("freeze_backbone", False)):     # 冻结部分强制 eval：BN 走 running stats，既不回传也不漂移
        for name, m in model.named_children():
            if name not in ("pool", "head"):
                m.eval()
    sampler.set_epoch(epoch)
    tau = float(cfg.get("tau", 0.1))
    q_size = int(cfg.get("queue_size", 0))
    log_interval = int(cfg.get("log_interval", 20))
    total_steps = len(train_loader)

    # ---- K 轮：可微几何验证模块的损失超参 ----
    geo_iter = iter(geo_loader) if geo_loader is not None else None
    g_temp = float(cfg.get("geo_temp", 0.05))
    g_sigma = float(cfg.get("geo_sigma", 0.08))
    g_margin = float(cfg.get("geo_margin", 0.3))
    lam_geo = float(cfg.get("lambda_geo", 1.0))
    geo_acc, geo_sp, geo_sn, geo_n = 0.0, 0.0, 0.0, 0

    # ---- 网格级对比（§6.2 第 2 条）----
    # 只在 spatial_head 模式下可用：此时 head 之前的池化向量就是 grid_ch × G × G 的展平结果，
    # 可按位置拆成 G² 个局部描述子（默认 8×8×32）。
    grid_on = bool(cfg.get("grid_contrast", False))
    g_grid = int(cfg.get("pool_grid", 8))
    grid_ch = int(cfg["embed_dim"]) // (g_grid * g_grid)
    lam_grid = float(cfg.get("lambda_grid", 0.5))
    tau_grid = float(cfg.get("tau_grid", 0.1))
    grid_acc, grid_n = 0.0, 0
    if grid_on:
        assert bool(cfg.get("spatial_head", False)), "grid_contrast 需要 spatial_head=true"

    acc, n, nonfinite = 0.0, 0, 0
    t_last = time.time()
    for step, batch in enumerate(train_loader, 1):
        img = batch["image"].to(device)
        labels = batch["label"].to(device)

        # 编码器内部整段强制 fp32（见 encoder.py 的 forward 注释）：AMP 在这条主干上会在个别批
        # 溢出成 NaN。对比损失本来就在 fp32 里算，这里不再套外层 autocast，保持数值一致。
        if grid_on:
            z_raw, pooled = model(img, return_pooled=True)
        else:
            z_raw = model(img)
        z = F.normalize(z_raw.float(), dim=1)                   # 对比损失在 fp32 上算

        has_q = queue_z is not None and queue_z.numel() > 0
        # 队列常驻 CPU（只当不回传梯度的负样本 key），用时才搬上 GPU：4096×2048 若常驻 GPU，
        # 每步都 torch.cat 出一个 34MB 的临时块，会明显加剧本就吃紧的显存碎片。
        loss_c = supcon_loss(z, labels,
                             queue_z.to(device) if has_q else None,
                             queue_labels.to(device) if has_q else None, tau=tau)
        loss_v = variance_loss(z)
        loss = loss_c + float(cfg.get("lambda_vicreg", 0.05)) * loss_v

        # 网格级对比：把 head 之前的 8×8 网格按位置拆开，逐位置做跨样本 SupCon。
        # 展平顺序是 (通道, 高, 宽)，因此必须先 reshape 成 (B, C, G, G) 再换到位置维，
        # 直接 reshape(B, G², C) 会得到错误的"位置"。
        loss_grid = None
        if grid_on:
            b = pooled.shape[0]
            cells = pooled.reshape(b, grid_ch, g_grid, g_grid).permute(0, 2, 3, 1)
            cells = F.normalize(cells.reshape(b, g_grid * g_grid, grid_ch).float(), dim=2)
            loss_grid = grid_supcon_loss(cells, labels, tau=tau_grid)
            loss = loss + lam_grid * loss_grid
            grid_acc += float(loss_grid) * b
            grid_n += b

        # 低层蒸馏（路线 A）：把 r8_norm 教师的判别结构压进主干。投影头只作用于主嵌入，
        # 检索路径不经过它 —— 学生仍有自由度超过教师（否则上限被人为压在 60.05%）。
        loss_d = None
        if proj is not None:
            t = lowres_teacher(img, int(cfg.get("distill_res", 8)))
            p = F.normalize(proj(z), dim=1)
            loss_d = (1.0 - (p * t).sum(dim=1)).mean()
            loss = loss + float(cfg.get("lambda_distill", 0.5)) * loss_d

        # 可微几何一致性（K 轮）：同一优化步内联合优化，梯度经【共享主干 conv0/scales[0]】回传，
        # 让"背景可被同一单应解释"直接塑形检索表征（不是外挂重排器，推理路径根本不走这条）。
        loss_g = None
        if geo_iter is not None:
            try:
                gb = next(geo_iter)
            except StopIteration:
                geo_iter = iter(geo_loader)
                gb = next(geo_iter)
            Fq = model.forward_local(gb["query"].to(device))     # (B,D,g,g) 逐位置 L2 归一化
            Fr = model.forward_local(gb["ref"].to(device))
            bq, d, g, _ = Fq.shape
            Fq = Fq.reshape(bq, d, g * g)                        # (B,D,N)
            Fr = Fr.reshape(bq, d, g * g)
            coords = grid_coords(g, Fq.device)
            s_pos = pair_score(Fq, Fr, coords, g_temp, g_sigma)  # 同源：干净真帧
            # 负对：参考逐个错位 -> 每个查询配到【同一批内其他所有源】的干净真帧（零额外前向）。
            # 只用单个错位时每个查询仅 1 个负样本，信号太薄；跑遍全部错位把负样本数提到 B×(B-1)。
            if bq < 2:
                s_neg = 1.0 - s_pos                              # 单样本批：无负样本可用，退化为 0 损失
            else:
                s_neg = torch.stack([pair_score(Fq, torch.roll(Fr, shifts=s, dims=0), coords,
                                                g_temp, g_sigma) for s in range(1, bq)]).mean(0)
            loss_g = geo_consistency_loss(s_pos, s_neg, g_margin)
            loss = loss + lam_geo * loss_g
            geo_acc += float(loss_g) * bq
            geo_sp += float(s_pos.mean()) * bq
            geo_sn += float(s_neg.mean()) * bq
            geo_n += bq

        # 坏批保护：整批丢弃，既不 backward 也不碰 GradScaler（否则 NaN 梯度会把 scale 拖到 0，
        # 训练静默空转 —— baseline 256px 全规格踩过这个坑）
        if not torch.isfinite(loss):
            nonfinite += 1
            if nonfinite == 1:
                print(f"[diag] 首个非有限损失：z finite={bool(torch.isfinite(z_raw).all())}, "
                      f"z_absmax={float(z_raw.abs().max()):.3g}, "
                      f"param_ok={all(bool(torch.isfinite(p).all()) for p in model.parameters())}",
                      flush=True)
            optimizer.zero_grad(set_to_none=True)
            if nonfinite % 50 == 0:
                print(f"  [e{epoch} {step}/{total_steps}] 非有限损失累计 {nonfinite} 步，已丢弃",
                      flush=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(cfg.get("clip_grad_norm", 5.0)))
        optimizer.step()
        scheduler.step()

        # ---- 跨批内存（XBM）入队：只存 detach 的嵌入与标签，不回传梯度（存 CPU，省显存碎片） ----
        if q_size > 0:
            new_z, new_l = z.detach().cpu(), labels.detach().cpu()
            if queue_z is None or queue_z.numel() == 0:
                queue_z, queue_labels = new_z, new_l
            else:
                queue_z = torch.cat([queue_z, new_z], dim=0)[-q_size:]
                queue_labels = torch.cat([queue_labels, new_l], dim=0)[-q_size:]

        bs = img.shape[0]
        acc += loss.item() * bs
        n += bs
        if step % log_interval == 0 or step == total_steps:
            now = time.time()
            ms = (now - t_last) / log_interval * 1000
            t_last = now
            std = torch.sqrt(z.var(dim=0) + 1e-4).mean().item()
            print(f"  [e{epoch} {step:>5}/{total_steps}] loss={loss.item():.4f} "
                  f"con={loss_c.item():.4f} var={loss_v.item():.4f} "
                  f"dis={(loss_d.item() if loss_d is not None else 0.0):.4f} "
                  f"geo={(loss_g.item() if loss_g is not None else 0.0):.4f} "
                  f"grid={(loss_grid.item() if loss_grid is not None else 0.0):.4f} "
                  f"| emb_std={std:.4f} q={0 if queue_z is None else queue_z.shape[0]} "
                  f"| lr={optimizer.param_groups[0]['lr']:.2e} nf={nonfinite} "
                  f"| {ms:.0f}ms/step eta={(total_steps - step) * ms / 60000:.1f}min", flush=True)
    print(f"  [epoch {epoch}] avg loss={acc / max(n, 1):.4f} | 样本 {n} | nf={nonfinite}")
    if geo_n:
        print(f"  [epoch {epoch} geo] 几何一致性 loss={geo_acc / geo_n:.4f} | "
              f"正对分数={geo_sp / geo_n:.3f} 负对分数={geo_sn / geo_n:.3f}（分离度 "
              f"{(geo_sp - geo_sn) / geo_n:.3f}）| 几何对 {geo_n}")
    if grid_n:
        print(f"  [epoch {epoch} grid] 网格级对比 loss={grid_acc / grid_n:.4f} | 样本 {grid_n}")
    if writer:
        writer.add_scalar("train/loss", acc / max(n, 1), epoch)
    return queue_z, queue_labels


def main():
    parser = argparse.ArgumentParser(description="AURA 自建单编码器 · 全图对比学习")
    parser.add_argument("--config", default="aura/configs/aura_full.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--limit_videos", type=int, default=None, help="冒烟：缩小 train/val/test 视频数")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="临时覆盖任意配置项，可重复")
    args = parser.parse_args()

    cfg = load_config(os.path.abspath(args.config))
    for k, v in (("epochs", args.epochs), ("steps_per_epoch", args.steps_per_epoch)):
        if v is not None:
            cfg[k] = v
    if args.limit_videos is not None:
        cfg["num_train_videos"] = args.limit_videos
        cfg["num_val_videos"] = max(args.limit_videos // 4, 1)
        cfg["num_test_videos"] = max(args.limit_videos // 4, 1)
    for kv in args.set:
        k, v = kv.split("=", 1)
        import yaml
        cfg[k] = yaml.safe_load(v)
        print(f"[override] {k} = {cfg[k]}")

    set_seed(int(cfg.get("seed", 42)))
    # set_seed 会把 cudnn.benchmark 打开（baseline 为提速设的）；但它会让 cuDNN 在首次前向
    # 自动调优卷积算法，瞬时分配超大 workspace —— 实测在 256px 全规格下把启动峰值从 ~7.5GB
    # 抬到 18.8GB（占卡 83%），直接触发 Windows 共享内存回退。本训练强制关掉，换回确定的显存占用。
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    device = get_device(cfg)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"device = {device} ({gpu}) | torch = {torch.__version__}")

    index = build_ffpp_index(cfg)
    train_ds, sampler, train_loader, val_ds, db_loader, query_loader, geo_loader = _build_loaders(
        index, cfg, int(cfg.get("seed", 42)))
    print(f"训练样本 {len(train_ds)} 张 / {sampler.n_groups} 个源 | "
          f"batch = {cfg['groups_per_batch']}源 × {cfg['samples_per_group']} = "
          f"{int(cfg['groups_per_batch']) * int(cfg['samples_per_group'])} | "
          f"step/epoch = {len(train_loader)} | 评估图库 {len(db_loader.dataset)} / 查询 {len(query_loader.dataset)}")
    if geo_loader is not None:
        print(f"[geo] 几何验证训练对 {len(geo_loader.dataset)} 个（训练集伪造帧）/ batch {cfg.get('geo_batch', 4)} "
              f"/ size {cfg.get('geo_size', cfg['image_size'])} | λ_geo={cfg.get('lambda_geo', 1.0)} "
              f"| temp={cfg.get('geo_temp', 0.05)} σ={cfg.get('geo_sigma', 0.08)}")

    model = SourceEncoder.build_from_config(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {n_param / 1e6:.2f} M | embed_dim={cfg['embed_dim']} image_size={cfg['image_size']}")

    # J 轮冻结实验：从已训好的权重初始化，再冻结 backbone（只训 pool+head）。
    # 动机：H/I 两轮都"e1 最高、之后回落"，说明 SupCon+XBM 的梯度在持续侵蚀空间头保存的低层布局。
    # 掐断 backbone 的梯度回传后，只让 head 学"如何把冻结的空间特征读成判别嵌入"，就不会再破坏它。
    if cfg.get("init_ckpt"):
        init_path = os.path.abspath(str(cfg["init_ckpt"]))
        # strict=False：K 轮在 I-best 之上新增了局部描述子头（local_head.*），旧权重里没有这些键，
        # 新键随机初始化即可（骨干完全加载）。
        state = torch.load(init_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[init] 从 {init_path} 载入权重（strict=False | 缺键 {len(missing)} / 多余键 {len(unexpected)}）")
    if bool(cfg.get("freeze_backbone", False)):
        for p in model.parameters():
            p.requires_grad_(False)
        for m in (model.pool, model.head):
            for p in m.parameters():
                p.requires_grad_(True)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[freeze] 冻结 backbone，仅训 pool+head：可训参数 {n_train / 1e6:.3f} M")

    # 低层蒸馏的投影头（路线 A）：仅训练期辅助分支，不进 checkpoint —— 评估脚本按 E 轮的
    # 结构加载主权重时不会遇到多余的键；检索路径也完全不走它。
    proj = None
    if bool(cfg.get("distill", False)):
        d_t, d_h = int(cfg.get("distill_dim", 192)), int(cfg.get("distill_hidden", 512))
        proj = torch.nn.Sequential(
            torch.nn.Linear(int(cfg["embed_dim"]), d_h), torch.nn.GELU(),
            torch.nn.Linear(d_h, d_t)).to(device)
        print(f"[distill] 教师={cfg.get('distill_teacher', 'r8_norm')}"
              f"（{cfg.get('distill_res', 8)}×{cfg.get('distill_res', 8)} 逐图标准化缩略图，{d_t} 维）"
              f" | λ_distill={cfg.get('lambda_distill', 0.5)}"
              f" | 投影头 {cfg['embed_dim']}→{d_h}→{d_t}"
              f"（{sum(p.numel() for p in proj.parameters()) / 1e3:.1f}K 参数，辅助分支，不进 checkpoint）")

    params = [p for p in model.parameters() if p.requires_grad] + (list(proj.parameters()) if proj is not None else [])
    optimizer = torch.optim.AdamW(params, lr=float(cfg["lr"]),
                                  betas=tuple(cfg.get("betas", [0.9, 0.999])),
                                  weight_decay=float(cfg.get("weight_decay", 1e-4)))
    epochs = int(cfg["epochs"])
    total_steps = max(1, epochs * len(train_loader))
    warmup = max(1, int(float(cfg.get("warmup_ratio", 0.05)) * total_steps))

    constant_lr = str(cfg.get("lr_schedule", "cosine")) == "constant"

    def lr_lambda(step):
        if constant_lr:                                    # J 轮：恒定 lr，不做余弦退火
            return 1.0
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_dir = ensure_dir(cfg.get("save_dir", "runs"))
    log_dir = ensure_dir(cfg.get("log_dir", "runs/tb"))
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    except Exception as e:                                       # noqa: BLE001 tensorboard 可选
        print(f"[warn] tensorboard 不可用，仅打印日志: {e}")
        writer = None

    history, queue_z, queue_labels = [], None, None
    best_t1, best_epoch = -1.0, 0   # best-epoch 跟踪：按 cosine 帧级 Top-1 取最优权重
    for epoch in range(1, epochs + 1):
        queue_z, queue_labels = train_one_epoch(
            model, sampler, train_loader, optimizer, scheduler, cfg, device,
            epoch, queue_z, queue_labels, proj, writer, geo_loader)
        ckpt = os.path.join(save_dir, f"{cfg.get('name', 'aura')}_e{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"  已保存: {ckpt}")

        if int(cfg.get("eval_every_epoch", 1)) and epoch % int(cfg.get("eval_every_epoch", 1)) == 0:
            res = evaluate(model, db_loader, query_loader, device)
            print(f"  [评估 e{epoch}]\n{format_report(res)}")
            for m in ("cosine", "hamming"):
                for k, v in res[m].items():
                    if k != "n":
                        writer and writer.add_scalar(f"eval/{m}_{k}", v, epoch)
            t1 = res["cosine"]["frame_top1"]
            if t1 > best_t1:                     # 刷新最佳 -> 另存 _best.pth（末轮可能已退化，见 I 轮动机）
                best_t1, best_epoch = t1, epoch
                best_ckpt = os.path.join(save_dir, f"{cfg.get('name', 'aura')}_best.pth")
                torch.save(model.state_dict(), best_ckpt)
                print(f"  [best] e{epoch} 余弦帧级 Top-1={t1 * 100:.2f}% 刷新最佳 -> {best_ckpt}")
            for label, line in KILL_LINES.items():
                print(f"  [kill 判据] {label} {line * 100:.2f}% -> "
                      f"{'通过' if t1 > line else '未过'}（本模型余弦帧级 Top-1 = {t1 * 100:.2f}%）")
            res["epoch"] = epoch
            res["best"] = (best_epoch == epoch)
            history.append(res)
            with open(os.path.join(save_dir, f"{cfg.get('name', 'aura')}_history.json"),
                      "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            # 退化鲁棒基准只在最后一轮跑：8000 查询 × 6 档 = 4.8 万次前向，约 70 分钟，放每轮太贵。
            if epoch == epochs and cfg.get("degradation_eval") == "last":
                print("  [退化评估] 图库干净、查询六档退化（约 70 分钟）...", flush=True)
                dres = evaluate_degradation(model, db_loader, query_loader.dataset.samples,
                                            cfg, device)
                print(f"  [退化 e{epoch}]\n{format_degradation(dres)}", flush=True)
                with open(os.path.join(save_dir, f"{cfg.get('name', 'aura')}_degradation.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(dres, f, indent=2, ensure_ascii=False)

    if device.type == "cuda":
        print(f"GPU 峰值显存: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
