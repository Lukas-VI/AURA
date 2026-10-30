"""路线 1b：自己训练【对应型局部描述子】，用已知随机几何变换做稠密自监督（InfoNCE）。

为什么需要
----------
路线 1 最小原型（_probe_diff_geom.py）证实：
    Q2 可微几何验证模块梯度能回传（✅），
    但直接复用整图对比学习编码器的 feature map 时 Q1 判别力不过（AUC≈0.39<随机，锐度×N≈17）
    —— 全局编码器的特征在空间上平滑、不为逐位置对应优化，导致软匹配弥散、单应解无效。
本脚本即补上缺失的一环：训练一个**以对应为目标**的局部描述子。

自监督信号（无需任何标注）
    对一张真图施加【已知】随机几何变换（旋转 ±deg / 缩放 / 平移）得到变形图；由已知变换可
    精确算出"原图某位置 -> 变形图对应位置"，于是可构造稠密对应监督：
        锚点 = 原图位置描述子；正样本 = 变形图在【对应位置】的描述子（grid_sample 取），
        负样本 = 变形图其余所有位置的描述子；用 InfoNCE 拉近正样本、推远负样本。
    这与测试集六档几何退化同源（旋转/缩放/平移/裁剪），因此训出的描述子天然对该类退化稳健。

数据用 FF++ train split 的真实帧（2400 张）；只在真图上训，描述子学的是"场景对应"这种通用能力。

用法（项目根目录 d:\py\AURA 下）
    $env:CUDA_DEVICE_ORDER="PCI_BUS_ID"; $env:CUDA_VISIBLE_DEVICES="1"; $env:PYTHONPATH="d:\py\AURA"
    冒烟： E:\conda\envs\pytorch\python.exe -u aura\runs\_train_local_desc.py --steps 100
    正式： E:\conda\envs\pytorch\python.exe -u aura\runs\_train_local_desc.py --steps 3000
"""
import argparse
import json
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aura.encoder import build_local_descriptor
from baseline.data.dataset import build_ffpp_index
from baseline.utils import get_device, load_config

DEFAULT_CFG = os.path.abspath("aura/configs/aura_full_spatial_i.yaml")
DEFAULT_OUT = "D:/py/AURA/aura/runs/local_desc.pth"


class RealFrames(Dataset):
    """train split 真实帧：读图 -> resize -> RGB -> [0,1] 张量（描述子训练用真图）。

    photo_prob > 0 时对【同一张图】先做一次随机光度退化，再交给几何变形分支。
    两个视图来自同一张退化图，对应关系不变，因此学的是"退化前后描述子仍能对应"——
    这正是查询端（JPEG/模糊/低分辨率）与图库端（干净）错配时需要的性质。
    """

    def __init__(self, samples, size: int, photo_prob: float = 0.0, photo_mode: str = "mixed"):
        self.samples = list(samples)
        self.size = int(size)
        self.photo_prob = float(photo_prob)
        self.photo_mode = photo_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        bgr = cv2.imread(self.samples[i]["path"], cv2.IMREAD_COLOR)
        bgr = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.photo_prob > 0 and random.random() < self.photo_prob:
            rgb = photo_degrade(rgb, self.photo_mode)
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)


def photo_degrade(rgb: np.ndarray, mode: str = "mixed") -> np.ndarray:
    """随机光度退化（与查询端档位同源）：JPEG / 高斯模糊 / 噪声 / 降采样 / 亮度对比度。

    mode="jpeg" 只做 JPEG，且质量限制在 15~45：目的是补 jpeg5 这一格的失效，同时避免
    把模糊/噪声也塞进来——上一轮 photo_prob=0.7（含全部退化）把 clean 拉低了 1.4pp。
    """
    h, w = rgb.shape[:2]
    if mode == "jpeg":
        ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(15, 45)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else rgb
    op = random.choice(("jpeg", "blur", "noise", "down", "bright"))
    if op == "jpeg":
        q = random.randint(5, 50)
        ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else rgb
    if op == "blur":
        return cv2.GaussianBlur(rgb, (0, 0), random.uniform(0.5, 3.0))
    if op == "noise":
        n = np.random.normal(0.0, random.uniform(2.0, 12.0), rgb.shape)
        return np.clip(rgb.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if op == "down":
        side = random.choice((64, 96, 128))
        small = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha, beta = random.uniform(0.7, 1.3), random.uniform(-25, 25)
    return np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def base_grid(h, w, device):
    """align_corners=False 口径的像素中心归一化坐标 (h,w,2)，(x,y)。"""
    ys = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device)
    xs = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)


def random_affine(b, device, max_deg, smin, smax, max_t):
    """随机仿射 M（2x3，归一化坐标，输入->输出）：旋转 ±max_deg、缩放 [smin,smax]、平移 ±max_t。"""
    ang = torch.empty(b, device=device).uniform_(-max_deg, max_deg) * math.pi / 180
    sc = torch.empty(b, device=device).uniform_(smin, smax)
    tx = torch.empty(b, device=device).uniform_(-max_t, max_t)
    ty = torch.empty(b, device=device).uniform_(-max_t, max_t)
    ca, sa = torch.cos(ang) * sc, torch.sin(ang) * sc
    A = torch.stack([torch.stack([ca, -sa], -1), torch.stack([sa, ca], -1)], 1)   # (B,2,2)
    t = torch.stack([tx, ty], 1).unsqueeze(2)                                    # (B,2,1)
    return A, t


def warp_with_affine(x, A, t):
    """按"输入->输出=M"的约定生成变形图 xw = grid_sample(x, theta_inv)，theta_inv 为 M 的逆。"""
    b = x.shape[0]
    Ainv = torch.linalg.inv(A)                       # (B,2,2)
    tinv = -torch.bmm(Ainv, t).squeeze(2)            # (B,2)
    theta = torch.cat([Ainv, tinv.unsqueeze(2)], dim=2)                          # (B,2,3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def dense_infonce(f0, f1, pos, valid, tau):
    """稠密 InfoNCE：锚点 f0 对【正样本】pos 与【负样本集】f1 全部位置做对比。

    f0/f1/pos: (B,D,N)；valid: (B,N)。logits = [s_pos, s_neg...]，正样本在索引 0。
    """
    sp = (f0 * pos).sum(1) / tau                       # (B,N) 正样本相似度
    sn = torch.einsum("bdn,bdm->bnm", f0, f1) / tau    # (B,N,N) 负样本相似度（含正所在格）
    logits = torch.cat([sp.unsqueeze(2), sn], dim=2)   # (B,N,1+N)
    loss = -F.log_softmax(logits, dim=2)[:, :, 0]      # (B,N) -log p(正样本)
    loss = (loss * valid).sum() / valid.sum().clamp_min(1.0)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3, help="stride=2 的层数；网格 = size / 2**depth")
    ap.add_argument("--backbone", default="tiny",
                    help="局部描述子主干：tiny = 自建 11K 网络；resnet18 = ImageNet 预训练浅层")
    ap.add_argument("--trunk_lr", type=float, default=0.0,
                    help="resnet18 主干的学习率；0 = 冻结主干只训 1×1 投影")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.1, help="InfoNCE 温度")
    ap.add_argument("--max_deg", type=float, default=15.0, help="随机旋转幅度（±度）")
    ap.add_argument("--smin", type=float, default=0.8)
    ap.add_argument("--smax", type=float, default=1.25)
    ap.add_argument("--max_t", type=float, default=0.15, help="随机平移幅度（归一化坐标）")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--photo_prob", type=float, default=0.0,
                    help="每张图先做随机光度退化的概率（JPEG/模糊/噪声/降采样/亮度）")
    ap.add_argument("--photo_mode", default="mixed", choices=["mixed", "jpeg"],
                    help="mixed=全部退化类型；jpeg=只做 JPEG(q15~45)，专补极端压缩档")
    ap.add_argument("--json", default="D:/py/AURA/aura/runs/local_desc_train.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    cfg = load_config(os.path.abspath(args.config))
    dev = get_device(cfg)
    index = build_ffpp_index(cfg)
    reals = [s for s in index["train"] if s["kind"] == "real"]
    stride = 4 if args.backbone == "resnet18" else 2 ** args.depth
    print(f"device={dev} | train reals {len(reals)} | size={args.size} | dim={args.dim} | "
          f"grid={args.size // stride}² | steps={args.steps} | bs={args.bs} | "
          f"backbone={args.backbone} | photo_prob={args.photo_prob}", flush=True)

    ds = RealFrames(reals, args.size, photo_prob=args.photo_prob, photo_mode=args.photo_mode)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)

    model = build_local_descriptor(args.backbone, dim=args.dim, width=args.width,
                                   depth=args.depth,
                                   train_trunk=args.trunk_lr > 0).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"描述子网络可训参数 {n_par / 1e3:.0f}K "
          f"（总 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M）", flush=True)
    head = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith(("conv1", "bn1", "layer1", "layer2"))]
    groups = [{"params": head, "lr": args.lr}]
    trunk = [p for n, p in model.named_parameters()
             if p.requires_grad and n.startswith(("conv1", "bn1", "layer1", "layer2"))]
    if trunk:
        groups.append({"params": trunk, "lr": args.trunk_lr})
    opt = torch.optim.Adam(groups, lr=args.lr)

    g = args.size // model.grid_stride
    coords = base_grid(g, g, dev)                       # (g,g,2) 原图位置（描述子网格中心）
    coords_flat = coords.reshape(-1, 2)                 # (N,2)

    model.train()
    step = 0
    losses, t0 = [], time.time()
    while step < args.steps:
        for x in loader:
            if step >= args.steps:
                break
            x = x.to(dev)
            b = x.shape[0]
            A, t = random_affine(b, dev, args.max_deg, args.smin, args.smax, args.max_t)
            xw = warp_with_affine(x, A, t)

            f0 = model(x)                               # (B,D,g,g)
            f1 = model(xw)
            # 已知对应：原图位置 p -> 变形图位置 M p
            out = torch.einsum("bij,nj->bni", A, coords_flat) + t.squeeze(2).unsqueeze(1)   # (B,N,2)
            valid = (out.abs() <= 1.0).all(-1).float()                                        # (B,N)
            pos = F.grid_sample(f1, out.unsqueeze(1), align_corners=False, padding_mode="zeros")  # (B,D,1,N)

            b_, d_, h_, w_ = f0.shape
            f0f = f0.reshape(b_, d_, -1)
            f1f = f1.reshape(b_, d_, -1)
            posf = pos.reshape(b_, d_, -1)
            loss = dense_infonce(f0f, f1f, posf, valid, args.tau)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
            step += 1
            if step % args.log_every == 0 or step == 1:
                print(f"  step {step:5d}/{args.steps} | loss {float(loss):.4f} | "
                      f"valid {valid.mean():.2f} | {(time.time() - t0):.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"[save] {args.out}")
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({"steps": args.steps, "bs": args.bs, "size": args.size, "dim": args.dim,
                   "depth": args.depth, "backbone": args.backbone, "tau": args.tau,
                   "loss_first": losses[0],
                   "loss_last": losses[-1], "loss_mean_last100": float(np.mean(losses[-100:])),
                   "params": n_par}, f, ensure_ascii=False, indent=2)
    print(f"[loss] 首步 {losses[0]:.4f} -> 末步 {losses[-1]:.4f} | 末 100 步均值 {np.mean(losses[-100:]):.4f}")


if __name__ == "__main__":
    main()
