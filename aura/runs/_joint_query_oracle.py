"""联合查询 oracle 上界分析（低成本前置验证）：r8_norm 零训练初筛 top-K -> I-best 深度嵌入重排。

背景与动机
----------
深度模型 I-best 单独 clean 帧级 Top-1 = 59.67%，并未超过零训练 r8_norm 的 60.05%
（差 0.38pp，噪声内）；J 轮冻结实验（55.46%）已证伪"梯度回传"假设。唯一还没验证的杠杆是
【把两者串起来】：用廉价的传统 CV（r8_norm，192 维确定性描述子）做初筛、拿 I-best 深度嵌入
在候选里重排。问题在于 —— 这样做到底有没有收益？

本脚本先不工程化完整流水线，只回答一个决定性问题：
    联合查询的【硬上界】= r8_norm 的 recall@K。若 recall@K 本身就压不过两个单独方法的上限
    （即正确答案根本不在 r8_norm 的候选里），则两者错误高度重叠、联合查询无杠杆，不值得做。
    反之再用 I-best 重排看能实际拿到多少。

逐档读数（clean + 6 个几何档，图库干净、只退化查询，与既有协议一致）
    档位 |  r8_norm 单独 T1/T5 |  I-best 单独 T1/T5  |  recall@K (K=1/5/10/20/50/100)  |  joint T1/T5

用法（项目根目录 d:\py\AURA 下）
    $env:PYTHONPATH="d:\py\AURA"
    E:\conda\envs\pytorch\python.exe -u aura\runs\_joint_query_oracle.py
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aura.data import GEOM_LEVELS, ContrastiveDataset
from aura.encoder import SourceEncoder
from baseline.data.dataset import build_ffpp_index
from baseline.utils import get_device, load_config

DEFAULT_CFG = os.path.abspath("aura/configs/aura_full_spatial_i.yaml")
DEFAULT_CKPT = "D:/py/AURA/aura/runs/aura_full_spatial_i_best.pth"
DEFAULT_OUT = "D:/py/AURA/aura/runs/joint_query_oracle.json"


def r8_norm(img, res=8):
    """零训练确定性描述子：res×res 缩略图 -> 逐图标准化 -> L2 归一化（192 维）。

    与 train.py 的 lowres_teacher / 扫描脚本的 r8_norm 同配方（仅维度排列不同，对逐图标准化 +
    L2 归一化等变）。这里从【已退化的查询张量】上算，保证与深度支路看到同一张图。
    """
    t = F.interpolate(img, size=(res, res), mode="area").reshape(img.shape[0], -1)
    t = (t - t.mean(dim=1, keepdim=True)) / (t.std(dim=1, keepdim=True) + 1e-6)
    return F.normalize(t, dim=1)


@torch.no_grad()
def encode_both(model, loader, dev):
    """一趟前向同时取两套表示：I-best 深度嵌入（2048，L2 归一化）与 r8_norm（192，L2 归一化）。"""
    model.eval()
    zs, ts, srcs = [], [], []
    for b in loader:
        img = b["image"].to(dev)
        zs.append(F.normalize(model(img).float(), dim=1).cpu())
        ts.append(r8_norm(img).cpu())
        srcs.extend(list(b["source_id"]))
    return torch.cat(zs), torch.cat(ts), np.array(srcs)


def t1_t5(order, db_src, q_src):
    """由按分排序的图库索引算帧级 Top-1 / Top-5 命中率。"""
    r = db_src[order.cpu().numpy()]
    hit = (r == q_src[:, None])
    return float(hit[:, :1].any(1).mean()), float(hit[:, :5].any(1).mean())


def recall_at(order, db_src, q_src, ks):
    """正确答案落在 top-K 候选里的比例（=联合查询硬上界）。"""
    r = db_src[order.cpu().numpy()]
    hit = (r == q_src[:, None])
    return {k: float(hit[:, :k].any(1).mean()) for k in ks}


def joint_topk(S_r8, S_deep, K, db_src, q_src):
    """r8_norm top-K 初筛 -> I-best 余弦在候选内重排 -> 帧级 Top-1/Top-5。"""
    cand = S_r8.topk(min(K, S_r8.shape[1]), dim=1).indices          # (Nq, K) r8 初筛候选
    sub = torch.gather(S_deep, 1, cand)                            # 候选内的深度相似度
    ranked = torch.gather(cand, 1, sub.argsort(dim=1, descending=True))
    return t1_t5(ranked, db_src, q_src)


def fuse(S_r8, S_deep, w):
    """另一种联合：逐查询 z-score 后把两路相似度线性融合（w 给 r8 的权重），全图库排序。

    用于回答"两个方法错误是否互补"—— 若融合能超过任一单独方法，说明有可用的联合增益
    （哪怕不是初筛+重排那种省算力的形式）。
    """
    def z(S):
        return (S - S.mean(dim=1, keepdim=True)) / (S.std(dim=1, keepdim=True) + 1e-6)
    return w * z(S_r8) + (1.0 - w) * z(S_deep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n_query", type=int, default=2000,
                    help="查询子采样数（确定性抽样，可复现）。默认 2000 约 17min；0 = 全量 8000（约 68min）。")
    ap.add_argument("--ks", default="1,5,10,20,50,100", help="recall@K 与联合重排的候选档位")
    ap.add_argument("--ws", default="0.2,0.35,0.5,0.65,0.8", help="融合时给 r8 的权重扫描")
    args = ap.parse_args()
    KS = [int(x) for x in args.ks.split(",")]
    WS = [float(x) for x in args.ws.split(",")]

    cfg = load_config(os.path.abspath(args.config))
    dev = get_device(cfg)
    print(f"device={dev} | config={args.config} | ckpt={args.ckpt}", flush=True)

    model = SourceEncoder.build_from_config(cfg).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()
    print(f"参数 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)

    index = build_ffpp_index(cfg)
    real = [s for s in index["test"] if s["kind"] == "real"]
    fake = [s for s in index["test"] if s["kind"] == "fake"]
    if args.n_query and args.n_query < len(fake):
        fake = random.Random(0).sample(fake, args.n_query)          # 固定种子 -> 各档位查询集一致
    size = int(cfg["image_size"])
    print(f"图库 {len(real)} / 查询 {len(fake)}", flush=True)

    # 图库只编码一次且保持干净
    db_loader = DataLoader(ContrastiveDataset(real, image_size=size, train=False, seed=0),
                           batch_size=args.bs, shuffle=False, num_workers=0)
    db_z, db_t, db_src = encode_both(model, db_loader, dev)
    print(f"[db] 深度 {tuple(db_z.shape)} | r8 {tuple(db_t.shape)}", flush=True)

    levels = ("clean",) + tuple(GEOM_LEVELS)
    t0 = time.time()
    out = {"n_db": len(db_src), "n_query": len(fake), "ks": KS, "levels": {}}
    print("\n" + "=" * 118)
    print(f"{'档位':>10} | {'r8_norm 单独':>18} | {'I-best 单独':>18} | "
          f"{'recall@K（联合硬上界）':>26} | {'joint T1':>16} | {'fusion 最佳':>14}")
    print("-" * 118)
    for lvl in levels:
        q_ds = ContrastiveDataset(fake, image_size=size, train=False, seed=0, degrade_level=lvl)
        q_loader = DataLoader(q_ds, batch_size=args.bs, shuffle=False, num_workers=0)
        q_z, q_t, q_src = encode_both(model, q_loader, dev)

        S_r8 = q_t @ db_t.t()
        S_deep = q_z @ db_z.t()

        r8_t1, r8_t5 = t1_t5(S_r8.topk(5, dim=1).indices, db_src, q_src)
        dp_t1, dp_t5 = t1_t5(S_deep.topk(5, dim=1).indices, db_src, q_src)
        rc = recall_at(S_r8.topk(max(KS), dim=1).indices, db_src, q_src, KS)
        joints = {k: joint_topk(S_r8, S_deep, k, db_src, q_src) for k in KS}
        fuses = {w: t1_t5(fuse(S_r8, S_deep, w).topk(5, dim=1).indices, db_src, q_src) for w in WS}
        bw = max(fuses, key=lambda w: fuses[w][0])

        print(f"{lvl:>10} | {r8_t1 * 100:7.2f}%/{r8_t5 * 100:6.2f}% | "
              f"{dp_t1 * 100:7.2f}%/{dp_t5 * 100:6.2f}% | "
              f"{' '.join(f'@{k}:{rc[k] * 100:.1f}' for k in KS)} | "
              f"{' '.join(f'@{k}:{joints[k][0] * 100:.1f}' for k in (10, 50, 100))} | "
              f"w={bw} {fuses[bw][0] * 100:.2f}%", flush=True)
        out["levels"][lvl] = {
            "r8_t1": r8_t1, "r8_t5": r8_t5, "deep_t1": dp_t1, "deep_t5": dp_t5,
            "recall": rc,
            "joint": {str(k): {"t1": joints[k][0], "t5": joints[k][1]} for k in KS},
            "fuse": {str(w): {"t1": fuses[w][0], "t5": fuses[w][1]} for w in WS},
            "fuse_best_w": bw, "fuse_best_t1": fuses[bw][0], "fuse_best_t5": fuses[bw][1],
        }
        del S_r8, S_deep, q_z, q_t
    print("=" * 118)
    print(f"[time] {(time.time() - t0) / 60:.1f} min", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
