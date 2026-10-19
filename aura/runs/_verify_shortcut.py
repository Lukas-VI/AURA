"""最小验证（零训练）：在已训好的深度嵌入旁边并一条 32x32 缩略图短路，
看帧级 Top-1 能否越过"零训练 tiny32 整图"基线 58.80%。

不训练、不改模型，只做一次编码：
    z_deep = L2norm(SourceEncoder(图))            # identity 轮 e6 权重，2048 维
    z_low  = L2norm(flatten(area_resize(图,32)))  # 3072 维，纯像素
两个子块各自 L2 归一化后，拼接嵌入 z=[w_a*z_a; w_b*z_b] 的余弦满足
    cos = (cos_a + lam*cos_b) / (1 + lam)，  lam = (w_b/w_a)^2
所以扫 lam 只需一次矩阵运算，不必重复前向。

判据：lam=0 应复现 identity 轮的帧级 47.75%；lam=inf 应复现 tiny32 的 ~58.80%。
只要中间某个 lam 超过 58.80%，就说明"低层信息是瓶颈"这个假设成立。

临时验证脚本，跑完即删。用法（项目根目录下）：
    $env:PYTHONPATH="d:\py\AURA"; E:\conda\envs\pytorch\python.exe -u aura\runs\_verify_shortcut.py
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aura.data import ContrastiveDataset
from aura.encoder import SourceEncoder
from baseline.data.dataset import build_ffpp_index
from baseline.utils import get_device, load_config

CFG = "aura/configs/aura_full.yaml"
CKPT = "D:/py/AURA/aura/runs/aura_full_e6.pth"     # identity 监督那轮的 e6
LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
K = 5


def build_loaders(cfg):
    size = int(cfg["image_size"])
    bs = int(cfg.get("eval_batch_size", 4))
    idx = build_ffpp_index(cfg)
    real = [s for s in idx["test"] if s["kind"] == "real"]
    fake = [s for s in idx["test"] if s["kind"] == "fake"]
    db = DataLoader(ContrastiveDataset(real, image_size=size, train=False, seed=0),
                    batch_size=bs, shuffle=False, num_workers=0)
    q = DataLoader(ContrastiveDataset(fake, image_size=size, train=False, seed=0),
                   batch_size=bs, shuffle=False, num_workers=0)
    return db, q


@torch.no_grad()
def encode(model, loader, device):
    """一次前向同时产出：深度嵌入、原样缩略图、逐样本标准化缩略图。"""
    deep, low_raw, low_std, ids, idents, srcs = [], [], [], [], [], []
    for batch in loader:
        x = batch["image"].to(device).float()
        z = F.normalize(model(x).float(), dim=1)                              # (B,2048)
        thumb = F.interpolate(x, size=(32, 32), mode="area").flatten(1)       # (B,3072) 原始像素
        low_raw.append(F.normalize(thumb, dim=1).cpu())
        m = thumb.mean(dim=1, keepdim=True)
        s = thumb.std(dim=1, keepdim=True) + 1e-6
        low_std.append(F.normalize((thumb - m) / s, dim=1).cpu())
        deep.append(z.cpu())
        ids += list(batch["id"])
        idents += list(batch["identity"])
        srcs += list(batch["source_id"])
    return (torch.cat(deep), torch.cat(low_raw), torch.cat(low_std),
            np.array(ids), np.array(idents), np.array(srcs))


def report(S, k, db_src, db_ident, q_src, q_ident):
    """给定相似度矩阵 (Q,DB)，返回帧级/视频级 Top-k 命中率。"""
    idx = S.topk(k, dim=1).indices.numpy()                     # (Q,k)
    hit_f = (db_src[idx] == q_src[:, None]).any(axis=1)
    hit_v = (db_ident[idx] == q_ident[:, None]).any(axis=1)
    return {f"frame_top{k}": float(hit_f.mean()), f"video_top{k}": float(hit_v.mean())}


def main():
    cfg = load_config(os.path.abspath(CFG))
    dev = get_device(cfg)
    print(f"device = {dev} | ckpt = {CKPT}", flush=True)

    model = SourceEncoder.build_from_config(cfg).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev))
    model.eval()
    print(f"参数 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)

    db_loader, q_loader = build_loaders(cfg)
    print("[encode] 图库…", flush=True)
    Zdb, Rdb, Tdb, db_ids, db_ident, db_src = encode(model, db_loader, dev)
    print("[encode] 查询…", flush=True)
    Zq, Rq, Tq, q_ids, q_ident, q_src = encode(model, q_loader, dev)
    print(f"图库 {len(db_ids)} / 查询 {len(q_ids)}", flush=True)

    S_deep = Zq @ Zdb.T
    S_raw = Rq @ Rdb.T
    S_std = Tq @ Tdb.T

    def line(tag, S):
        r1 = report(S, 1, db_src, db_ident, q_src, q_ident)
        r5 = report(S, K, db_src, db_ident, q_src, q_ident)
        print(f"  {tag:>26s} | frame_top1={r1['frame_top1'] * 100:6.2f}% "
              f"frame_top5={r5['frame_top5'] * 100:6.2f}% "
              f"video_top1={r1['video_top1'] * 100:6.2f}% "
              f"video_top5={r5['video_top5'] * 100:6.2f}%", flush=True)

    print("\n=== 单路对照（应复现已知基线） ===", flush=True)
    line("仅深度嵌入 (lam=0)", S_deep)
    line("仅原样缩略图 (lam=inf)", S_raw)
    line("仅标准化缩略图 (lam=inf)", S_std)

    for tag, S_low in (("原样缩略图", S_raw), ("标准化缩略图", S_std)):
        print(f"\n=== 拼接扫权重 · 低层={tag} ===", flush=True)
        for lam in LAMS:
            S = (S_deep + lam * S_low) / (1.0 + lam)
            line(f"lam={lam:g}", S)

    print("\n判据：任一行 frame_top1 > 58.80% 即确认低层信息是瓶颈。", flush=True)


if __name__ == "__main__":
    main()
