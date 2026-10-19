"""免训练对照基线：冻结特征 / 感知哈希 / 缩略图 —— 判断"自造哈希网络"是否真有价值。

背景
----
AURA 当前基线（HashVAE 复现）在同一测试协议下的读数是：背景码 Top-1 = 24.28%、
Top-5 = 64.43%（随机基线 Top-1 = 0.0625%）。但这个数字**不能**直接说明哈希网络学到了
东西：FF++ 的伪造帧 `X_Y` 与它的源真实帧 `original/X` 是**同一帧序号**（见
data/dataset.py 的 `frame_map.get(fr)`），背景区域几乎逐像素相同，任务本质是近重复检索。

因此本脚本用一组**完全不训练**的检索器做对照。判据很直接：
    若某个免训练方法已能匹敌甚至超过哈希网络 -> 自造哈希网络的增量价值不成立。

对照方法与区域变体
------------------
    方法    tiny    32x32 缩略图，L2
            phash   灰度 DCT 感知哈希（64 位），汉明
            hist    8x8x8 RGB 颜色直方图，L1
            dino    冻结视觉特征（DINOv2 ViT-S/14 CLS，回退 ResNet18），余弦
    区域    full    整图
            bg      前景置零（保留背景，对应"背景是来源指纹"假设）
            fg      背景置零（保留前景）

每种方法 × 每种区域都报告 **帧级** 与 **视频级** 的 Top-1 / Top-5。
帧级命中 = 检索到 source_id（同一帧序号的源真实帧）；视频级命中 = 检索到同一源视频的任意帧。

运行（项目根目录下）
--------------------
    E:\\conda\\envs\\pytorch\\python.exe -m baseline.analysis.untrained_baselines --config configs/baseline.yaml
    # 冒烟：--limit 8 只取少量查询与图库
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from ..data.dataset import build_ffpp_index
from ..data.preprocessing import FaceParseMask
from ..utils import ensure_dir, load_config, resolve_path

WORK = 256          # 工作分辨率（与 256px 掩膜缓存一致）
SMALL = 32          # 缩略图 / 感知哈希边长
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# 本地 dinov2 仓库（torch.hub 走 GitHub API 会触发未认证限流，故用本地目录加载）
_DINO_LOCAL = os.path.join(os.environ.get("TEMP", ""), "aura_dinov2", "repo", "dinov2-main")


# ---------------- 图像 / 区域 ----------------
def _load_rgb(path, size=WORK):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _apply_region(rgb, mask, region):
    """按区域变体屏蔽掉不需要的部分：'bg' 只留背景，'fg' 只留前景。"""
    if region == "full" or mask is None:
        return rgb
    m = mask[..., None].astype(np.float32)
    keep = (1.0 - m) if region == "bg" else m
    return (rgb.astype(np.float32) * keep).astype(np.uint8)


# ---------------- 免训练描述子 ----------------
def _tiny(rgb):
    x = cv2.resize(rgb, (SMALL, SMALL), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return x.reshape(-1)


def _phash(rgb, n=8):
    """灰度 DCT 感知哈希：取左上 n×n 低频，> 中位数记 1，得到 n*n 位 0/1 码。"""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (SMALL, SMALL), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:n, :n].reshape(-1)
    return (d > np.median(d[1:])).astype(np.float32)     # 去掉 DC 项再取中位数


def _hist(rgb, bins=8):
    """8x8x8 RGB 颜色直方图（L1 归一化）；对场景光照/颜色极敏感，是背景匹配的强基线。"""
    q = (rgb.astype(np.int32) * bins // 256).clip(0, bins - 1)
    idx = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]
    h = np.bincount(idx.reshape(-1), minlength=bins ** 3).astype(np.float32)
    return h / (h.sum() + 1e-8)


def _load_feature_model(device):
    """加载冻结视觉特征模型：优先本地 DINOv2，失败回退 torchvision ResNet18。"""
    if os.path.isdir(_DINO_LOCAL):
        try:
            m = torch.hub.load(_DINO_LOCAL, "dinov2_vits14", source="local", verbose=False)
            m.eval().to(device)
            for p in m.parameters():
                p.requires_grad_(False)
            print("[feat] 冻结特征 = DINOv2 ViT-S/14（384 维 CLS）")
            return ("dino", m, 384)
        except Exception as e:                        # noqa: BLE001 失败即回退
            print(f"[feat] DINOv2 加载失败（{e}），回退 ResNet18")
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    print("[feat] 冻结特征 = torchvision ResNet18（512 维池化特征）")
    return ("resnet18", m, 512)


@torch.no_grad()
def _feat_preprocess(rgbs, device, size=224):
    """RGB uint8 列表 -> 归一化张量（DINOv2 与 ResNet18 共用同一套 ImageNet 预处理）。"""
    arr = np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA) for x in rgbs])
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2))).to(device)


# ---------------- 检索 ----------------
@torch.no_grad()
def _topk(q, db, k, q_chunk=128, db_chunk=1024):
    """分块 L2 距离 Top-K；对二值码/直方图等价于汉明/L1 的排序（仅排序，尺度无所谓）。"""
    q, db = q.float(), db.float()
    out_d, out_i = [], []
    for s in range(0, q.shape[0], q_chunk):
        qi = q[s:s + q_chunk]
        row_d, row_i = [], []
        for t in range(0, db.shape[0], db_chunk):
            dj = db[t:t + db_chunk]
            dist = torch.cdist(qi, dj, p=2)
            td, ti = dist.topk(min(k, dj.shape[0]), largest=False, dim=1)
            row_d.append(td)
            row_i.append(ti + t)
        row_d = torch.cat(row_d, dim=1)
        row_i = torch.cat(row_i, dim=1)
        td, ti = row_d.topk(k, largest=False, dim=1)
        out_d.append(td)
        out_i.append(row_i.gather(1, ti))
    return torch.cat(out_d, 0), torch.cat(out_i, 0)


def _score(query, db, pair, k=(1, 5)):
    """计算帧级 / 视频级 Top-K 命中率。pair = (source_ids, db_ids, db_videos, q_videos)。"""
    _, idx = _topk(query, db, max(k))
    src_ids, db_ids, db_videos, q_videos = pair
    frame_hit = {t: 0 for t in k}
    video_hit = {t: 0 for t in k}
    for i, src in enumerate(src_ids):
        got = idx[i].tolist()
        for t in k:
            top = got[:t]
            if src in [db_ids[j] for j in top]:
                frame_hit[t] += 1
            if q_videos[i] in [db_videos[j] for j in top]:
                video_hit[t] += 1
    n = max(len(src_ids), 1)
    res = {f"frame_top{t}": frame_hit[t] / n for t in k}
    res.update({f"video_top{t}": video_hit[t] / n for t in k})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只取前 N 个查询/图库做冒烟")
    ap.add_argument("--methods", default="tiny,phash,hist,dino")
    ap.add_argument("--regions", default="full,bg,fg")
    ap.add_argument("--out", default="results/untrained_baselines.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = build_ffpp_index(cfg)
    samples = index[args.split]
    gallery = [s for s in samples if s["kind"] == "real"]
    query = [s for s in samples if s["kind"] == "fake"]
    if args.limit > 0:
        gallery, query = gallery[:args.limit], query[:args.limit]
    print(f"[data] {args.split}：图库(真实)={len(gallery)} 查询(伪造)={len(query)}"
          f" | 随机基线 Top-1={1/max(len(gallery),1):.6f}")

    # 掩膜一律按工作分辨率 WORK 生成：128 的掩膜上采样到 256 会把边界糊掉，
    # 直接影响 bg/fg 区域变体的质量。256 掩膜若未缓存会现场解析（GPU 上每张几十毫秒）。
    mask_size = WORK
    splitter = FaceParseMask(mask_size, cache_dir=ensure_dir(cfg.get("mask_cache_dir", "cache/masks")))

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    need_feat = "dino" in methods
    feat_model = _load_feature_model(device) if need_feat else None

    # 一次性遍历图库 + 查询，逐区域抽取 tiny/phash/hist，并批量抽冻结特征
    items = [(s["path"], False, s) for s in gallery] + [(s["path"], True, s) for s in query]
    n_all = len(items)
    dims = {"tiny": SMALL * SMALL * 3, "phash": 64, "hist": 8 ** 3}
    desc = {m: {r: np.zeros((n_all, dims[m]), np.float32) for r in regions}
            for m in ("tiny", "phash", "hist")}
    feats = {r: np.zeros((n_all, feat_model[2]), np.float32) for r in regions} if need_feat else {}
    feat_buf = {r: [] for r in regions} if need_feat else {}
    feat_idx = []

    def _flush_feat():
        if not feat_idx:
            return
        for r in regions:
            x = _feat_preprocess(feat_buf[r], device)
            out = feat_model[1](x)
            v = out.detach().cpu().numpy().astype(np.float32)
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)     # 余弦
            feats[r][feat_idx] = v
            feat_buf[r].clear()
        feat_idx.clear()

    for i, (path, _is_q, _s) in enumerate(items):
        rgb = _load_rgb(path, WORK)
        mask = splitter.mask_for_path(path, mask_size)
        if mask.shape[0] != WORK:                                      # 掩膜按 image_size 缓存，缩放回工作分辨率
            mask = cv2.resize(mask, (WORK, WORK), interpolation=cv2.INTER_AREA)
        for r in regions:
            rr = _apply_region(rgb, mask, r)
            desc["tiny"][r][i] = _tiny(rr)
            desc["phash"][r][i] = _phash(rr)
            desc["hist"][r][i] = _hist(rr)
        if need_feat:
            for r in regions:
                feat_buf[r].append(_apply_region(rgb, mask, r))
            feat_idx.append(i)
            if len(feat_idx) >= 32:
                _flush_feat()
        if (i + 1) % 500 == 0 or i + 1 == n_all:
            print(f"  [desc] {i+1}/{n_all}", flush=True)
    if need_feat:
        _flush_feat()
    splitter.unload()

    ng = len(gallery)
    q_ids = [s["source_id"] for s in query]
    db_ids = [s["id"] for s in gallery]
    db_videos = [s["identity"] for s in gallery]
    q_videos = [s["identity"] for s in query]
    pair = (q_ids, db_ids, db_videos, q_videos)

    results = {"n_gallery": ng, "n_query": len(query), "split": args.split,
               "random_top1": 1.0 / max(ng, 1), "methods": {}}
    print("\n" + "=" * 96)
    print(f"{'方法':<10}{'区域':<6}{'帧Top-1':>10}{'帧Top-5':>10}{'视频Top-1':>11}{'视频Top-5':>11}")
    print("-" * 96)
    for m in methods:
        block = {}
        for r in regions:
            if m in ("tiny", "phash", "hist"):
                qd = torch.from_numpy(desc[m][r][ng:])
                dd = torch.from_numpy(desc[m][r][:ng])
            else:
                qd = torch.from_numpy(feats[r][ng:])
                dd = torch.from_numpy(feats[r][:ng])
            sc = _score(qd, dd, pair)
            block[r] = sc
            print(f"{m:<10}{r:<6}{sc['frame_top1']*100:>9.2f}%{sc['frame_top5']*100:>9.2f}%"
                  f"{sc['video_top1']*100:>10.2f}%{sc['video_top5']*100:>10.2f}%")
        results["methods"][m] = block
    print("=" * 96)
    print("参照：HashVAE 基线（背景码） 帧Top-1=24.28% 帧Top-5=64.43%")

    out = resolve_path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
