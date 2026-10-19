"""几何不变局部匹配快速原型：初筛 top-K -> SIFT/ORB + RANSAC 几何验证重排。

背景与动机
----------
现有深度编码器（I-best）在几何退化下崩盘：clean 59.67% -> rot10 32.71% / crop0.8 21.70%
/ crop0.7off 7.98%。GAP 只对【全局平移】近似不变，一旦旋转/裁剪就整体错位。而检索的判别
信号（背景低频全局色调）恰恰被几何类退化破坏。

本原型验证一条与传统 CV 正交的路线：把"细粒度重排"交给【局部特征几何验证】——
    (1) 用便宜描述子（r8_norm / 深度嵌入 / 二者融合）做初筛拿 top-K 候选；
    (2) 对每个候选做 SIFT 互最近邻 + Lowe ratio，再用 cv2.findHomography(RANSAC) 数内点；
    (3) 以内点数为分重排。
SIFT 的尺度/旋转/平移不变性正好对上测试集的六档几何扰动；RANSAC 用背景几何一致性把"抄近路"
的散点匹配过滤掉。

关键前置事实（来自 joint_query_oracle.json，全量 8000）：
r8_norm 的 recall@100 在 rot5/rot10/shift8/zoom1.15 都有 ~99%，crop0.8 94.6%，crop0.7off 68.4%。
即【初筛不是瓶颈，重排才是】—— clean 下 recall@10=98.4% 但联合重排仅 ~60%。首轮主实验
（300 查询, K=150）已证实：几何重排把六档 T1 全部抬到 ~70%，重排条件正确率（真值在候选内时
被排到 top-1 的比例）在所有档位都 ≈65~73% 且与档位无关 —— 说明残差损失全落在初筛召回上。

本脚本在首轮基础上扩展两个验证（第二、三轮）：
    1) K 扫描：候选按初筛分排序，top-k 天然嵌套（top-150 ⊂ top-300 ⊂ ...），故只按最大 K 算一次
       匹配分，再对多个 k 嵌套评估 —— 几乎零额外算力，用来看 crop0.7off 的初筛天花板能否抬升。
    2) 初筛变体：--prefilter {r8,deep,fuse}。deep = I-best 深度嵌入余弦；fuse = 逐查询 z-score 后
       线性融合。用于检验"深度嵌入初筛 + SIFT 重排"是否既提初筛、又保住深度模型的贡献。

协议与既有一致：图库 = test split 1600 真实帧（干净、只编码一次）；查询 = test split 伪造帧
（只退化查询）。命中判定 = 检索到精确 source_id（X_f帧号）。

判据（用于决定是否推进到"诊断+方案"合并论文）：
    rot10 > ~45%  /  crop0.8 > ~35%  /  crop0.7off > ~25%  才算有戏（现基线分别 32.71/21.70/7.98）。

用法（项目根目录 d:\py\AURA 下）
    冒烟： $env:PYTHONPATH="d:\py\AURA"; E:\conda\envs\pytorch\python.exe -u aura\runs\_probe_geom_invariant.py --n_query 30 --prefilter_k 20
    主实验： ... --n_query 300 --prefilter_k 150
    K 扫描： ... --n_query 300 --prefilter_ks 100,200,300,500,800 --levels crop0.7off,crop0.8
    混合：   ... --n_query 300 --prefilter_k 150 --prefilter fuse
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np

from aura.data import GEOM_LEVELS, ContrastiveDataset, apply_degradation
from baseline.data.dataset import build_ffpp_index
from baseline.utils import get_device, load_config

DEFAULT_CFG = os.path.abspath("aura/configs/aura_full_spatial_i.yaml")
DEFAULT_CKPT = "D:/py/AURA/aura/runs/aura_full_spatial_i_best.pth"   # I-best 深度编码器权重
DEFAULT_OUT = "D:/py/AURA/aura/runs/geom_invariant_probe.json"


def r8_norm_img(rgb: np.ndarray, res: int = 8) -> np.ndarray:
    """零训练确定性描述子：res×res 缩略图 -> 逐图标准化 -> L2 归一化（192 维）。

    与 _joint_query_oracle.py 的 r8_norm 同配方，只是这里在 numpy 上算（原型纯 CPU，无需模型）。
    """
    t = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    t = (t - t.mean()) / (t.std() + 1e-6)
    return t / (np.linalg.norm(t) + 1e-6)


def load_pipeline(path: str, size: int, level: str) -> np.ndarray:
    """与 ContrastiveDataset 完全同口径读图：BGR 读入 -> resize 到 size -> BGR2RGB -> 确定性退化。"""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if level:
        rgb = apply_degradation(rgb, level)
    return np.ascontiguousarray(rgb)


def detector(kind: str):
    return cv2.SIFT_create() if kind == "sift" else cv2.ORB_create(nfeatures=1000)


def make_matcher(kind: str):
    norm = cv2.NORM_L2 if kind == "sift" else cv2.NORM_HAMMING
    return cv2.BFMatcher(norm, crossCheck=False)


def features(det, rgb: np.ndarray, max_kp: int):
    """在灰度图上提特征；max_kp>0 时按 response 取最强的 max_kp 个（控内存与匹配耗时）。"""
    kp, desc = det.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    if desc is None or len(kp) == 0:
        return [], None
    if max_kp and len(kp) > max_kp:
        order = np.argsort([-k.response for k in kp])[:max_kp]
        kp = [kp[i] for i in order]
        desc = desc[order]
    return kp, desc


def match_score(matcher, q_kp, q_desc, c_kp, c_desc, ratio: float, ransac_thresh: float, min_good: int):
    """查询特征 对 候选特征 做 knnMatch -> Lowe ratio 过滤 -> RANSAC homography 数内点。

    返回 (good 数, 内点数)。内点数作为重排主分：它度量"背景几何一致"的匹配量，能滤掉换脸
    人脸区带来的散点匹配。少于 min_good 个好匹配则不做 RANSAC（解不稳）。
    """
    if q_desc is None or c_desc is None or len(q_desc) < 2 or len(c_desc) < 2:
        return 0, 0
    knn = matcher.knnMatch(q_desc, c_desc, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < min_good:
        return len(good), 0
    src = np.float32([q_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([c_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    inl = int(mask.sum()) if mask is not None else 0
    return len(good), inl


def zscore(S: np.ndarray) -> np.ndarray:
    """逐查询（按行）z-score，用于把两路不同量纲的相似度拉到可加。"""
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-6)


def t1_t5_from_order(ranked_src: np.ndarray, q_src: np.ndarray):
    """给定每行"已按分排序"的图库 source_id，算帧级 Top-1 / Top-5 命中率。"""
    hit = ranked_src == q_src[:, None]
    return float(hit[:, :1].any(1).mean()), float(hit[:, :5].any(1).mean())


def encode_deep(model, samples, size, level, device, batch_size, seed):
    """用 ContrastiveDataset + 模型取 L2 归一化深度嵌入（与 load_pipeline 同图片口径）。"""
    import torch
    from aura.eval import encode_loader
    from torch.utils.data import DataLoader
    ds = ContrastiveDataset(samples, image_size=size, train=False, seed=seed, degrade_level=level)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    z, _, _, _ = encode_loader(model, loader, device)
    return z.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="I-best 权重（仅 --prefilter deep/fuse 时需要）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n_query", type=int, default=300,
                    help="查询子采样数（固定种子，可复现）。0 = 全量 8000。")
    ap.add_argument("--prefilter_k", type=int, default=150,
                    help="单一 K（未给 --prefilter_ks 时生效）：初筛候选数，=重排范围，recall@K 为硬上界")
    ap.add_argument("--prefilter_ks", default="",
                    help="逗号分隔的多个 K（如 100,200,300,500,800）：只按最大 K 算一次匹配，嵌套评估")
    ap.add_argument("--prefilter", default="r8", choices=["r8", "deep", "fuse"],
                    help="初筛打分方式：r8=零训练描述子；deep=I-best 深度嵌入；fuse=逐查询 z-score 融合")
    ap.add_argument("--w_r8", type=float, default=0.5, help="fuse 给 r8 的权重（deep 权重 = 1-该值）")
    ap.add_argument("--detector", default="sift", choices=["sift", "orb"])
    ap.add_argument("--max_kp", type=int, default=1000, help="每图最多保留的特征点数（按 response），0=不限")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio 阈值")
    ap.add_argument("--ransac_thresh", type=float, default=5.0, help="RANSAC 重投影阈值（像素）")
    ap.add_argument("--min_good", type=int, default=4, help="触发 RANSAC 的最少好匹配数")
    ap.add_argument("--levels", default="", help="逗号分隔档位子集；空=clean+全部 GEOM_LEVELS")
    args = ap.parse_args()

    cfg = load_config(os.path.abspath(args.config))
    size = int(cfg["image_size"])
    levels = [s for s in args.levels.split(",") if s] if args.levels else ["clean", *GEOM_LEVELS]
    KS = [int(x) for x in args.prefilter_ks.split(",") if x] or [args.prefilter_k]
    KMAX = max(KS)

    index = build_ffpp_index(cfg)
    real = [s for s in index["test"] if s["kind"] == "real"]
    fake = [s for s in index["test"] if s["kind"] == "fake"]
    if args.n_query and args.n_query < len(fake):
        fake = random.Random(0).sample(fake, args.n_query)          # 固定种子 -> 各档查询集一致

    det = detector(args.detector)
    matcher = make_matcher(args.detector)
    device = None
    model = None
    if args.prefilter in ("deep", "fuse"):
        import torch
        from aura.encoder import SourceEncoder
        device = get_device(cfg)
        model = SourceEncoder.build_from_config(cfg).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()
        print(f"深度编码器已加载：{args.ckpt}（{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, {device}）", flush=True)

    print(f"detector={args.detector} | size={size} | 图库 {len(real)} / 查询 {len(fake)} "
          f"| prefilter={args.prefilter} | K={KS} | max_kp={args.max_kp} | ratio={args.ratio}", flush=True)

    # ---------- 图库：只处理一次（干净）----------
    t0 = time.time()
    db_kp, db_desc, db_r8, db_src = [], [], [], []
    for i, s in enumerate(real):
        rgb = load_pipeline(s["path"], size, None)              # 图库保持干净
        kp, desc = features(det, rgb, args.max_kp)
        db_kp.append(kp)
        db_desc.append(desc)
        db_r8.append(r8_norm_img(rgb))
        db_src.append(s["source_id"])
        if (i + 1) % 400 == 0:
            print(f"  [db] {i + 1}/{len(real)}  ({(time.time() - t0):.0f}s)", flush=True)
    db_r8 = np.stack(db_r8)                                     # (Ndb, 192)
    db_src = np.array(db_src)
    db_z = encode_deep(model, real, size, None, device, int(cfg.get("eval_batch_size", 4)), 0) \
        if model is not None else None
    n_kp = np.array([len(k) for k in db_kp])
    print(f"[db] 完成 {len(real)} 张，特征点 均值 {n_kp.mean():.0f} 中位 {int(np.median(n_kp))} "
          f"（{time.time() - t0:.0f}s）", flush=True)

    # ---------- 逐档评测 ----------
    out = {"n_db": len(db_src), "n_query": len(fake), "prefilter": args.prefilter,
           "ks": KS, "detector": args.detector, "size": size, "levels": {}}
    print("\n" + "=" * 128)
    print(f"{'档位':>10} | {'K':>5} | {'初筛单独T1/T5':>17} | {'recall@K':>9} | "
          f"{'几何重排T1':>11} {'T5':>8} | {'重排@1|候选内':>13} | {'good/内点':>11}")
    print("-" * 128)
    texp = time.time()
    for lvl in levels:
        q_kp_list, q_desc_list, q_r8, q_src = [], [], [], []
        for s in fake:
            rgb = load_pipeline(s["path"], size, lvl)           # 只退化查询
            kp, desc = features(det, rgb, args.max_kp)
            q_kp_list.append(kp)
            q_desc_list.append(desc)
            q_r8.append(r8_norm_img(rgb))
            q_src.append(s["source_id"])
        q_r8 = np.stack(q_r8)
        q_src = np.array(q_src)
        q_z = encode_deep(model, fake, size, lvl, device, int(cfg.get("eval_batch_size", 4)), 0) \
            if model is not None else None

        # ---------- 初筛打分 ----------
        S_r8 = q_r8 @ db_r8.T                                   # (Nq, Ndb) 余弦（已 L2 归一化，点积即余弦）
        if args.prefilter == "r8":
            S = S_r8
        else:
            S_deep = q_z @ db_z.T
            S = S_deep if args.prefilter == "deep" \
                else args.w_r8 * zscore(S_r8) + (1.0 - args.w_r8) * zscore(S_deep)
        cand = np.argsort(-S, axis=1)[:, :KMAX]                 # (Nq, KMAX) 候选，按初筛分降序（嵌套）
        # 初筛单独 T1/T5（无论哪种初筛，都报当前初筛自己的读数）
        alone_t1, alone_t5 = t1_t5_from_order(db_src[np.argsort(-S, axis=1)[:, :5]], q_src)

        # ---------- 几何重排：只按 KMAX 算一次匹配分，再对每个 k 嵌套评估 ----------
        all_scores = []
        for i in range(len(q_src)):
            per = []
            for j in cand[i]:
                per.append(match_score(matcher, q_kp_list[i], q_desc_list[i],
                                       db_kp[j], db_desc[j], args.ratio, args.ransac_thresh, args.min_good))
            all_scores.append(per)
            if (i + 1) % 200 == 0:
                print(f"    [{lvl}] {i + 1}/{len(q_src)}  ({(time.time() - texp) / 60:.1f}min)", flush=True)

        lvl_res = {"prefilter_alone_top1": float(alone_t1), "prefilter_alone_top5": float(alone_t5), "by_k": {}}
        for k in KS:
            hit1 = hit5 = in_cand = in_cand_hit1 = 0
            for i in range(len(q_src)):
                scores = all_scores[i][:k]
                order = sorted(range(len(scores)), key=lambda t: (-scores[t][0], -scores[t][1]))
                ranked_src = db_src[cand[i, :k][order]]
                if q_src[i] in ranked_src[:1]:
                    hit1 += 1
                if q_src[i] in ranked_src[:5]:
                    hit5 += 1
                if q_src[i] in db_src[cand[i, :k]]:
                    in_cand += 1
                    if q_src[i] in ranked_src[:1]:
                        in_cand_hit1 += 1
            n = len(q_src)
            gm = float(np.mean([np.median([g for _, g in all_scores[i][:k]]) for i in range(n)]))
            im = float(np.mean([np.median([inl for inl, _ in all_scores[i][:k]]) for i in range(n)]))
            print(f"{lvl:>10} | {k:>5} | {alone_t1 * 100:7.2f}%/{alone_t5 * 100:6.2f}% | "
                  f"{in_cand / n * 100:8.2f}% | {hit1 / n * 100:10.2f}% {hit5 / n * 100:7.2f}% | "
                  f"{in_cand_hit1 / max(in_cand, 1) * 100:12.2f}% | {gm:5.0f}/{im:5.2f}", flush=True)
            lvl_res["by_k"][str(k)] = {
                "recall_k": in_cand / n,
                "geom_top1": hit1 / n, "geom_top5": hit5 / n,
                "cond_top1_given_in_cand": in_cand_hit1 / max(in_cand, 1),
                "good_median": gm, "inlier_median": im,
            }
        out["levels"][lvl] = lvl_res
        del q_kp_list, q_desc_list, q_r8, all_scores
    print("=" * 128)
    print(f"[time] 总 {(time.time() - texp) / 60:.1f} min", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
