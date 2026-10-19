"""路线 1 最小原型：把"几何验证原则"做成【可微模块】（不做 SIFT 可微，而是学到式几何一致性）。

动机
----
经典管线（_probe_geom_invariant.py）已证实：r8/fuse 初筛 + SIFT+RANSAC 重排能把六档几何
退化 T1 从 8~33% 抬到 60~72%。但 SIFT（DoG 极值/NMS/量化）与 RANSAC（随机采样+硬 inlier）
都不可微，无法端到端嵌进网络。本原型验证【可微替代】的可行性，只回答三个决定性小问题：

    Q1 判别力：这个可微几何一致性分数，能否把"真值对"和"随机负对"分开？
    Q2 可回传：梯度能否顺利回传到特征提取器（conv0+scales[0]）？梯度是否非零/非 NaN？
    Q3 可训练：只用这个分数构造的 hinge 损失，几步 SGD 能否把 loss 压下去？

若三项都过，则"可微几何一致性模块"作为论文的方案升级/完整闭环成立。

设计（全链可微，不碰任何不可微算子）
------------------------------------
    1) 特征图：取编码器早期 feature map F∈(B,C,H,W)（conv0 -> scales[0]），自适应池化到 G×G 网格。
    2) 密集匹配：通道 L2 归一化后算相关体 (B,G²,G²)，用【软互最近邻】权重 wa*wb（softmax 替代
       硬 argmax + Lowe ratio）得到逐对匹配置信度。
    3) 可微 DLT：以匹配置信度为权重（乘 sqrt(w)）构 DLT 方程 A h = b（固定 h33=1，避免零解），
       用 torch.linalg.lstsq 解单应 h —— 最小二乘本身可微。
    4) 软 inlier：把 a 点经 H 变换后与 b 点比较，残差 r 的软内点 s=exp(-r²/2σ²)；
       分数 = 加权软内点占比 Σ(w·s)/Σw ∈[0,1]。
    （软阈值 sigmoid/exp 替代 RANSAC 的硬判定；软 softmax 替代硬互最近邻。整链无离散算子。）

数据对构造（免检索循环，最小化）
    正对：伪造帧 query（按档位退化）与其【真实源帧】(samples 的 source_path，干净)。
    负对：同一 query 对【随机另一张真实帧】。
    两对共用同一特征提取器与同一可微模块。

用法（项目根目录 d:\py\AURA 下）
    $env:CUDA_DEVICE_ORDER="PCI_BUS_ID"; $env:CUDA_VISIBLE_DEVICES="1"; $env:PYTHONPATH="d:\py\AURA"
    E:\conda\envs\pytorch\python.exe -u aura\runs\_probe_diff_geom.py --n_pairs 64 --steps 20
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from aura.data import GEOM_LEVELS, apply_degradation
from aura.encoder import LocalDescriptor, SourceEncoder
from baseline.data.dataset import build_ffpp_index
from baseline.utils import get_device, load_config

DEFAULT_CFG = os.path.abspath("aura/configs/aura_full_spatial_i.yaml")
DEFAULT_CKPT = "D:/py/AURA/aura/runs/aura_full_spatial_i_best.pth"
DEFAULT_OUT = "D:/py/AURA/aura/runs/diff_geom_probe.json"


def load_rgb(path: str, size: int, level: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if level:
        rgb = apply_degradation(rgb, level)
    return np.ascontiguousarray(rgb)


def to_tensor(imgs, device):
    arr = np.stack(imgs)                                    # (B,H,W,3) uint8
    t = torch.from_numpy(arr.transpose(0, 3, 1, 2)).float().div_(255.0)
    return t.to(device)


def featmap(model, x, grid):
    """取早期特征图（conv0 -> scales[0]），池化到 G×G 网格并 L2 归一化通道。

    刻意不走 model.forward（含 AFF 池化/head，且 AFF 在 256px 下是显存大头）；只用主干前两层，
    足以验证"可微几何模块"能否回传梯度到特征提取器，且原型足够轻。返回 (B,C,G*G)。
    """
    x = model.conv0(x.float())
    x = model.scales[0](x)                                  # (B, base_channels, H, W)
    x = F.adaptive_avg_pool2d(x, grid)                      # (B, C, G, G)
    b, c, g, _ = x.shape
    x = x.reshape(b, c, g * g)                              # (B, C, N)
    return F.normalize(x, dim=1)                            # 逐位置（列）L2 归一化


def grid_coords(grid, device):
    """G×G 网格点的归一化坐标 [-1,1]（(x,y) 交错与 flatten 顺序一致：列优先 → k = j*G+i）。"""
    us = (torch.arange(grid, device=device).float() + 0.5) / grid * 2 - 1
    vs = (torch.arange(grid, device=device).float() + 0.5) / grid * 2 - 1
    x = us[None, :].repeat(grid, 1).reshape(-1)             # i 快速变化
    y = vs[:, None].repeat(1, grid).reshape(-1)
    return torch.stack([x, y], dim=1)                       # (N,2)


def soft_mutual_match(Fa, Fb, temp):
    """软互最近邻权重：wa=softmax_b(相关体), wb=softmax_a(相关体), w=wa*wb。返回 w (B,N,N)。"""
    corr = torch.bmm(Fa.transpose(1, 2), Fb)                # (B,N,N)
    wa = torch.softmax(corr / temp, dim=2)                  # 每个 a 在 b 上的分布
    wb = torch.softmax(corr / temp, dim=1)                  # 每个 b 在 a 上的分布
    return wa * wb


def diff_dlt(a_pts, b_pts, weights):
    """加权可微 DLT（固定 h33=1）：A(2N,8)->最小二乘解 h(8)。a_pts,b_pts (B,N,2)，w (B,N,N)。

    w 取"行最大值"（每个 a 的最佳 b）作为该对应点的置信度，避免把 N² 个弱匹配全塞进方程。
    """
    w = weights.max(dim=2).values.clamp_min(1e-6)           # (B,N) 每个 a 点的匹配置信度
    x, y = a_pts[..., 0], a_pts[..., 1]
    xp, yp = b_pts[..., 0], b_pts[..., 1]
    z = torch.zeros_like(x)
    one = torch.ones_like(x)
    # 行1: [x, y, 1, 0,0,0, -x*xp, -y*xp] = xp ; 行2: [0,0,0, x, y, 1, -x*yp, -y*yp] = yp
    # 推导：xp = (h11 x + h12 y + h13) / (h31 x + h32 y + 1)
    #   => h11 x + h12 y + h13 - xp·(h31 x + h32 y) = xp。（符号务必与 RHS=xp 一致，
    #      否则等价于在拟合"映射到 -b"的单应，自配对也会给出近零分。）
    r1 = torch.stack([x, y, one, z, z, z, -x * xp, -y * xp], dim=2)
    r2 = torch.stack([z, z, z, x, y, one, -x * yp, -y * yp], dim=2)
    # 关键：stack 的轴必须插在"系数列维(8)"之前，才能得到逐点 [r1_N, r2_N] 的行序。
    # 用 dim=3 会得到 (B,N,8,2)、reshape 后按"列"交错，矩阵退化（rank 7）、连恒等都不成立。
    A = torch.stack([r1, r2], dim=2).reshape(a_pts.shape[0], -1, 8)   # (B,2N,8)
    t = torch.stack([xp, yp], dim=2).reshape(a_pts.shape[0], -1)      # (B,2N)
    sw = torch.sqrt(w).repeat_interleave(2, dim=1).unsqueeze(2)       # 行权重 sqrt(w)
    sol = torch.linalg.lstsq(A * sw, (t * sw.squeeze(2)).unsqueeze(2)).solution  # (B,8,1)
    return sol.squeeze(2)                                             # (B,8)


def geom_score(h, a_pts, b_pts, weights, sigma):
    """软内点分数：a 点经 h 变换后与 b 点的残差 r，软内点 s=exp(-r²/2σ²)，加权占比作分。

    也可理解为"最佳匹配点对上的几何一致性程度"：背景若整体被同一单应解释，分数就高；
    换脸人脸区/散点匹配会拉高残差把分数压低 —— 这正是 RANSAC 要做的事，这里用软阈值替代。
    """
    x, y = a_pts[..., 0], a_pts[..., 1]
    xp, yp = b_pts[..., 0], b_pts[..., 1]
    denom = (h[:, 6:7] * x + h[:, 7:8] * y + 1.0).clamp_min(1e-6)
    xt = (h[:, 0:1] * x + h[:, 1:2] * y + h[:, 2:3]) / denom
    yt = (h[:, 3:4] * x + h[:, 4:5] * y + h[:, 5:6]) / denom
    r2 = (xt - xp) ** 2 + (yt - yp) ** 2
    s = torch.exp(-r2 / (2 * sigma ** 2))                   # 软内点
    w = weights.max(dim=2).values.clamp_min(1e-6)           # (B,N)
    return (w * s).sum(1) / w.sum(1)                        # (B,) ∈ (0,1)


def auc(pos, neg):
    """正负分数对的 AUC（Mann-Whitney）：随机取一对 pos/neg，pos>neg 的比例。"""
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n) + 0.5 * (p == n)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n_pairs", type=int, default=64, help="正/负对各自的样本数")
    ap.add_argument("--size", type=int, default=128, help="原型用较小输入以提速（模块在特征图上工作）")
    ap.add_argument("--grid", type=int, default=24, help="特征图池化网格 G（关键点 = G²）")
    ap.add_argument("--features", default="encoder", choices=["encoder", "local"],
                    help="特征来源：encoder=整图对比编码器早期 feature map；local=自己训练的局部描述子")
    ap.add_argument("--desc_ckpt", default="D:/py/AURA/aura/runs/local_desc.pth",
                    help="--features local 时的局部描述子权重（由 _train_local_desc.py 产出）")
    ap.add_argument("--desc_dim", type=int, default=32)
    ap.add_argument("--desc_width", type=int, default=32)
    ap.add_argument("--desc_depth", type=int, default=3, help="网格 = size / 2**desc_depth")
    ap.add_argument("--levels", default="clean,crop0.8,crop0.7off,zoom1.15,rot5,rot10,shift8")
    ap.add_argument("--temp", type=float, default=0.05, help="软匹配 softmax 温度")
    ap.add_argument("--sigma", type=float, default=0.08, help="软内点尺度（归一化坐标）")
    ap.add_argument("--steps", type=int, default=20, help="训练验证的 SGD 步数（只调 conv0+scales[0]）")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.3)
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    cfg = load_config(os.path.abspath(args.config))
    dev = get_device(cfg)
    model = SourceEncoder.build_from_config(cfg).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()
    print(f"device={dev} | ckpt={args.ckpt} | size={args.size} | grid={args.grid} | "
          f"n_pairs={args.n_pairs} | levels={args.levels}", flush=True)

    loc = None
    if args.features == "local":
        g = args.size // (2 ** args.desc_depth)              # 局部描述子网格由输入尺寸决定
        if g != args.grid:
            print(f"[info] 局部描述子网格 {g} != --grid {args.grid}，已强制 grid={g}", flush=True)
            args.grid = g
        loc = LocalDescriptor(dim=args.desc_dim, width=args.desc_width, depth=args.desc_depth).to(dev)
        loc.load_state_dict(torch.load(args.desc_ckpt, map_location=dev))
        loc.eval()
        print(f"局部描述子已加载：{args.desc_ckpt}（dim={args.desc_dim}, depth={args.desc_depth}）", flush=True)

    def extract(x):
        """按 --features 取逐位置特征 (B,C,N)：local=训练好的描述子；encoder=早期 feature map。"""
        if loc is not None:
            f = loc(x)
            b, c, h, w = f.shape
            return f.reshape(b, c, h * w)
        return featmap(model, x, args.grid)

    index = build_ffpp_index(cfg)
    reals = [s for s in index["test"] if s["kind"] == "real"]
    fakes = [s for s in index["test"] if s["kind"] == "fake"]
    fakes = [s for s in fakes if s.get("source_path")]          # 需要能拿到真实源帧
    rng = random.Random(0)
    fakes = rng.sample(fakes, min(args.n_pairs, len(fakes)))
    real_paths = [s["path"] for s in reals]
    # 预先解码缓存：候选/负样本反复用到真实帧
    coords = None

    levels = [x for x in args.levels.split(",") if x]
    out = {"n_pairs": len(fakes), "size": args.size, "grid": args.grid,
           "temp": args.temp, "sigma": args.sigma, "levels": {}}
    print("\n" + "=" * 96)
    print(f"{'档位':>10} | {'正对分数':>9} {'负对分数':>9} | {'AUC':>7} | "
          f"{'阈值0.5准确率':>13} | {'梯度范数 conv0/scales0':>22}")
    print("-" * 96)
    for lvl in levels:
        q_imgs = [load_rgb(s["path"], args.size, lvl) for s in fakes]
        p_imgs = [load_rgb(s["source_path"], args.size, None) for s in fakes]
        n_imgs = []
        for s in fakes:
            while True:
                cand = rng.choice(real_paths)
                if os.path.abspath(cand) != os.path.abspath(s["source_path"]):
                    break
            n_imgs.append(load_rgb(cand, args.size, None))

        with torch.no_grad():
            Fq = extract(to_tensor(q_imgs, dev))
            Fp = extract(to_tensor(p_imgs, dev))
            Fn = extract(to_tensor(n_imgs, dev))
            coords = grid_coords(args.grid, dev)
            wpos = soft_mutual_match(Fq, Fp, args.temp)
            # 匹配锐度：每个 a 的 "top-1 权重占比"，×N 后 >1 表示确实有峰（否则≈1 = 弥散/均匀）
            sharp = float((wpos.max(dim=2).values / wpos.sum(dim=2).clamp_min(1e-6)).mean()) * wpos.shape[1]
            score_pos = _pair_score(Fq, Fp, coords, args)
            score_neg = _pair_score(Fq, Fn, coords, args)
            self_score = float(_pair_score(Fq, Fq, coords, args).mean())   # 自配对：应为≈1（恒等变换）
            score_pos, score_neg = score_pos.cpu().numpy(), score_neg.cpu().numpy()

        a = auc(score_pos, score_neg)
        acc = float(((score_pos > 0.5).mean() + (score_neg <= 0.5).mean()) / 2)
        print(f"{lvl:>10} | {score_pos.mean():9.3f} {score_neg.mean():9.3f} | {a:7.3f} | "
              f"{acc * 100:12.1f}% | 锐度×N {sharp:5.2f} | 自配对 {self_score:.3f}", flush=True)
        out["levels"][lvl] = {"score_pos_mean": float(score_pos.mean()),
                              "score_neg_mean": float(score_neg.mean()), "auc": a, "acc": acc,
                              "match_sharp_xN": sharp, "self_score": self_score}

    # ---------- Q2/Q3：梯度回传 + 几步 SGD 能否压低 loss ----------
    print("\n" + "-" * 96)
    for p in model.parameters():
        p.requires_grad_(False)
    if loc is not None:
        for p in loc.parameters():
            p.requires_grad_(True)
        trainable = list(loc.parameters())
        tr_name = "局部描述子"
    else:
        trainable = list(model.conv0.parameters()) + list(model.scales[0].parameters())
        for p in trainable:
            p.requires_grad_(True)
        tr_name = "conv0+scales[0]"
    opt = torch.optim.Adam(trainable, lr=args.lr)
    print(f"[Q2/Q3] 只训练 {tr_name}（{sum(p.numel() for p in trainable) / 1e3:.0f}K 参数），"
          f"{args.steps} 步，lr={args.lr}", flush=True)
    lvl = "rot10" if "rot10" in levels else levels[0]
    tr_n = min(12, len(fakes))
    # 训练态批尺寸：encoder 分支激活太大曾 OOM 故降到 96px；local 描述子很小，用原尺寸以对齐网格
    tr_size = args.size if loc is not None else min(args.size, 96)
    q_imgs = [load_rgb(s["path"], tr_size, lvl) for s in fakes[:tr_n]]
    p_imgs = [load_rgb(s["source_path"], tr_size, None) for s in fakes[:tr_n]]
    n_imgs = [load_rgb(rng.choice(real_paths), tr_size, None) for s in fakes[:tr_n]]
    qs, ps, ns = to_tensor(q_imgs, dev), to_tensor(p_imgs, dev), to_tensor(n_imgs, dev)
    losses = []
    for step in range(args.steps):
        model.train()
        if loc is not None:
            loc.train()
        opt.zero_grad(set_to_none=True)
        Fq, Fp, Fn = extract(qs), extract(ps), extract(ns)
        cds = grid_coords(args.grid, dev)
        sp = _pair_score(Fq, Fp, cds, args)
        sn = _pair_score(Fq, Fn, cds, args)
        loss = F.relu(args.margin + sn - sp).mean()
        loss.backward()
        g_conv = trainable[0].grad
        g_norm = float(g_conv.norm()) if g_conv is not None else 0.0
        opt.step()
        losses.append(float(loss))
        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"  step {step:3d} | loss {float(loss):.4f} | grad_norm {g_norm:.3e}", flush=True)
    print("-" * 96)
    print(f"[Q2] 梯度非零且有限：{tr_name} 首参梯度范数 {g_norm:.3e}（>0 且非 NaN 即可回传）")
    print(f"[Q3] loss: 首步 {losses[0]:.4f} -> 末步 {losses[-1]:.4f}  "
          f"({'下降 ✓' if losses[-1] < losses[0] else '未下降'})")
    out["grad_norm_conv0"] = g_norm
    out["loss_first"] = losses[0]
    out["loss_last"] = losses[-1]
    out["loss_curve"] = losses

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[save] {args.out}")


def _pair_score(Fa, Fb, coords, args):
    """一对特征图 -> 几何一致性分数 (B,)。

    软互最近邻给权重 w (B,N,N)；把每个 a 点在 b 上的匹配求【软期望坐标】b_hat_i=Σ_j w_ij·coord_b_j，
    于是得到"软对应点"(coord_a_i, b_hat_i) 与置信度；再走加权可微 DLT 解单应、算软内点占比。
    注意：query 与候选在几何退化下并未对齐，对应关系全靠 w 建立，不能用同一套坐标冒充 b 点。
    """
    b = Fa.shape[0]
    c = coords.unsqueeze(0).expand(b, -1, -1)               # (B,N,2) query 网格坐标
    w = soft_mutual_match(Fa, Fb, args.temp)               # (B,N,N)
    wnorm = w / w.sum(dim=2, keepdim=True).clamp_min(1e-6)
    b_hat = torch.bmm(wnorm, c)                            # (B,N,2) 每个 a 的软期望匹配点
    h = diff_dlt(c, b_hat, w)
    return geom_score(h, c, b_hat, w, args.sigma)


if __name__ == "__main__":
    main()
