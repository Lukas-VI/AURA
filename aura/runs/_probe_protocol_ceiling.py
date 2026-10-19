"""零训练诊断：测本协议下【帧级 Top-1 的内在精度上限】。

背景
----
自建模型五轮全卡在 47±1%（低于零训练 32x32 整图基线 58.80%）。要判断病因是
    (a) 架构丢掉空间信息（head 前的 GAP 是 65536:1 满射，有救），还是
    (b) 协议本身在低层匹配下已接近饱和（58.80% 已离上限不远，应改口径），
就用一个**零训练低层匹配器**随【分辨率】与【区域】扫一遍，看上限落在哪。

方法
----
对每张测试图取 RGB(256) 与 BiSeNet 人脸掩膜（**只读磁盘缓存的 128×128 .npy**，
绝不触发解释网络，故无 GPU/模型开销），按
    region ∈ {full, bg(人脸置零), fg(背景置零)}
    res    ∈ {32, 64, 128}
算像素描述子  d = flatten(area_resize(rgb ⊙ region_mask, res))，L2 归一化后余弦检索。
图库 = 测试集真实帧(1600)，查询 = 测试集伪造帧(8000)，帧级命中 = 精确命中 source_id。

判读
----
bg 档随 res 上升：若 bg@128 显著高于 58.80%（比如 80%+）→ 协议没饱和、空间细节是瓶颈
→ 走 (a) 去 GAP；若 bg@128 与 58.80% 同量级封顶 → 协议饱和 → 走 (b) 改口径。

用法（项目根目录 d:\py\AURA 下）：
    $env:PYTHONPATH="d:\py\AURA"; E:\conda\envs\pytorch\python.exe -u aura\runs\_probe_protocol_ceiling.py
"""
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from aura.data import DEGRADATION_LEVELS, GEOM_LEVELS, apply_degradation, load_face_mask
from baseline.data.dataset import build_ffpp_index
from baseline.utils import load_config

CFG = os.path.abspath("aura/configs/aura_full.yaml")
WORK = 256                                    # 读图/掩膜的工作分辨率（FF++ 抽帧本身就是 256）
KS = (1, 5)
# 扫描配置：(区域, 分辨率, 描述子模式)
#   region: full 整图 / bg 人脸置零只留背景 / fg 背景置零只留人脸
#   mode:   raw  原始像素（复现已知读数）
#           norm 逐图全局标准化（去整体色调/对比度漂移）
#           hp   高通（减高斯低频，只留结构，去低频色调）
#           grid 4x4 分格各自去均值除标准差（惩罚局部错位，等价于空间网格描述子）
CONFIGS = [
    ("full", 32, "raw"), ("full", 128, "raw"),                 # sanity：应复现 58.71% 附近
    ("full", 32, "norm"), ("full", 64, "norm"), ("full", 128, "norm"),
    ("full", 128, "hp"), ("full", 128, "grid"),
    ("bg", 32, "norm"), ("bg", 128, "norm"), ("bg", 128, "grid"),
]
# 退化对照（与模型 §3.8 同协议）：图库干净、查询逐档退化，只测最强的低层描述子
# 方案 B 追加 GEOM_LEVELS：几何扰动（平移/旋转/裁剪/放大）——预期原始像素描述子会崩
DEG_LEVELS = tuple(DEGRADATION_LEVELS) + tuple(GEOM_LEVELS)
DEG_DESCS = [("full", 32, "raw"), ("full", 128, "raw")]
OUT = "D:/py/AURA/aura/runs/probe_protocol_ceiling.json"

# ---------- 手工描述子扫描（零训练）：回答"传统 CV 能否超过 58.71%" ----------
# 既有扫描只覆盖 RGB + 32/64/128px + 4 种归一化。这里补上此前未探索的轴：
#   更低分辨率(4/8/16/24)、显式低通(高斯 σ=4/8/16)后降采样、色彩空间(HSV/LAB/YCbCr)、
#   全局色相+饱和度直方图、灰度 DCT 低频块(J8/J16)、分块色彩矩(均值+方差)、
#   多尺度低频谱拼接、分路晚融合(相似度矩阵加权、网格搜 α)、PCA 白化(在图库上拟合)。
# 判读：若任一变体帧级 Top-1 明显越过 58.71%，则"协议已饱和"结论被推翻，
#       该变体即为新的 kill 线 + 路线 A 的蒸馏教师；否则饱和坐实，叙事更强。
CLS_NAMES = [
    "r4_raw", "r8_raw", "r16_raw", "r24_raw", "r32_raw", "r64_raw",   # 分辨率趋势
    "r8_norm", "r16_norm", "r24_norm", "r32_norm",                    # 逐图标准化
    "r32_bgnorm", "r32_fgnorm", "r16_bgnorm",                         # 区域（供晚融合）
    "blur4_r32_norm", "blur8_r32_norm", "blur16_r32_norm",            # 显式只留低频
    "hsv_r32_norm", "lab_r32_norm", "labab_r32_norm",                 # 色彩空间
    "ycc_r32_norm", "ycccbcr_r32_norm",
    "gridms4_r32", "gridms8_r32",                                     # 分块色彩矩
    "pyr_8_16_32", "huehsv_r64", "dct8_gray", "dct16_gray",           # 多尺度/直方图/DCT
]
FUSE_NAMES = ("r32_raw", "r32_norm", "r32_bgnorm", "r32_fgnorm", "r16_bgnorm", "huehsv_r64")
FUSE_PAIRS = [
    ("r32_fgnorm", "r32_bgnorm"),      # 人脸 × 背景（两者单独都低于整图，可能互补）
    ("r32_raw", "huehsv_r64"),         # 缩略图 × 颜色直方图（两者都 ~58%，误差若不重叠可叠加）
    ("r32_norm", "huehsv_r64"),
    ("r16_bgnorm", "r32_fgnorm"),
    ("r32_raw", "r32_norm"),
]
FUSE_ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def load_all(items, cache_dir):
    """一次性把所有图读成 (N,256,256,3) uint8 RGB + (N,256,256) uint8 掩膜，常驻内存。"""
    n = len(items)
    rgb_all = np.zeros((n, WORK, WORK, 3), np.uint8)
    mask_all = np.zeros((n, WORK, WORK), np.uint8)
    miss = 0
    t0 = time.time()
    for i, (path, _) in enumerate(items):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        if img.shape[:2] != (WORK, WORK):
            img = cv2.resize(img, (WORK, WORK), interpolation=cv2.INTER_AREA)
        rgb_all[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = load_face_mask(path, WORK, cache_dir)      # 缓存未命中返回 None
        if m is None:
            miss += 1
        else:
            mask_all[i] = np.clip(m * 255.0 + 0.5, 0, 255).astype(np.uint8)
        if (i + 1) % 1000 == 0 or i + 1 == n:
            print(f"  [load] {i+1}/{n}  掩膜缺失={miss}  {time.time()-t0:.0f}s", flush=True)
    print(f"[load] {n} 张完成，掩膜缺失 {miss}/{n}，用时 {time.time()-t0:.0f}s", flush=True)
    return rgb_all, mask_all


def descriptors(rgb_all, mask_all, region, res, mode="raw"):
    """按区域/分辨率/模式算像素描述子 (N, res*res*3)。"""
    out = np.empty((rgb_all.shape[0], res * res * 3), np.float32)
    for i in range(rgb_all.shape[0]):
        img = rgb_all[i]
        if res != WORK:
            img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        f = img.astype(np.float32)
        if region != "full":
            m = cv2.resize(mask_all[i], (res, res), interpolation=cv2.INTER_AREA)
            keep = ((255.0 - m) if region == "bg" else m)[..., None].astype(np.float32) / 255.0
            f = f * keep
        if mode == "norm":                                  # 去整体色调/对比度漂移
            f = (f - f.mean()) / (f.std() + 1e-6)
        elif mode == "hp":                                  # 减低频，只留结构
            f = f - cv2.GaussianBlur(f, (0, 0), 1.5)
        elif mode == "grid":                                # 4x4 分格各自标准化：惩罚局部错位
            g = 4
            c = res // g
            cells = np.ascontiguousarray(f).reshape(g, c, g, c, 3).transpose(0, 2, 1, 3, 4)
            cells = cells.reshape(g * g, c, c, 3)
            mu = cells.reshape(cells.shape[0], -1, 3).mean(axis=1)[:, None, None, :]
            sd = cells.reshape(cells.shape[0], -1, 3).std(axis=1)[:, None, None, :] + 1e-6
            f = (cells - mu) / sd
        out[i] = np.ascontiguousarray(f).reshape(-1)
    return out


_CLS_CACHE: dict = {}


def _res_u8(rgb_all, res):
    """按分辨率缓存降采样后的 uint8 RGB（内存便宜，避免重复 resize）。"""
    key = ("res", res)
    if key not in _CLS_CACHE:
        if res == WORK:
            _CLS_CACHE[key] = rgb_all
        else:
            out = np.empty((rgb_all.shape[0], res, res, 3), np.uint8)
            for i in range(rgb_all.shape[0]):
                out[i] = cv2.resize(rgb_all[i], (res, res), interpolation=cv2.INTER_AREA)
            _CLS_CACHE[key] = out
    return _CLS_CACHE[key]


def _mask_u8(mask_all, res):
    key = ("mask", res)
    if key not in _CLS_CACHE:
        if res == WORK:
            _CLS_CACHE[key] = mask_all
        else:
            out = np.empty((mask_all.shape[0], res, res), np.uint8)
            for i in range(mask_all.shape[0]):
                out[i] = cv2.resize(mask_all[i], (res, res), interpolation=cv2.INTER_AREA)
            _CLS_CACHE[key] = out
    return _CLS_CACHE[key]


def _std_all(f):
    """逐图全局标准化（去整体色调/对比度漂移），f 为 (N,H,W,C) float32。"""
    flat = f.reshape(f.shape[0], -1)
    mu = flat.mean(axis=1)[:, None, None, None]
    sd = flat.std(axis=1)[:, None, None, None] + 1e-6
    return (f - mu) / sd


def _cvt_all(rgb_u8, space):
    code = {"hsv": cv2.COLOR_RGB2HSV, "lab": cv2.COLOR_RGB2LAB, "ycc": cv2.COLOR_RGB2YCrCb}[space]
    out = np.empty_like(rgb_u8)
    for i in range(rgb_u8.shape[0]):
        out[i] = cv2.cvtColor(rgb_u8[i], code)
    return out


def _block_ms(f, g):
    """g×g 分块色彩矩：每块每通道的均值与标准差拼接（空间布局的粗粒度编码）。"""
    n, h, w, c = f.shape
    ch, cw = h // g, w // g
    cells = f[:, :ch * g, :cw * g].reshape(n, g, ch, g, cw, c).transpose(0, 1, 3, 2, 4, 5)
    cells = cells.reshape(n, g * g, ch * cw, c)
    return np.concatenate([cells.mean(axis=2), cells.std(axis=2)], axis=2).reshape(n, -1)


def _hsv_hist(rgb_u8, bins=32):
    """全局色相 + 饱和度直方图（各 bins 档，拼成 2*bins 维）。"""
    out = np.zeros((rgb_u8.shape[0], 2 * bins), np.float32)
    per = 256 // bins
    for i in range(rgb_u8.shape[0]):
        hsv = cv2.cvtColor(rgb_u8[i], cv2.COLOR_RGB2HSV)
        hh = np.bincount(hsv[..., 0].ravel(), minlength=256)[:bins * per].reshape(bins, per).sum(1)
        ss = np.bincount(hsv[..., 1].ravel(), minlength=256)[:bins * per].reshape(bins, per).sum(1)
        v = np.concatenate([hh, ss]).astype(np.float32)
        out[i] = v / (v.sum() + 1e-6)
    return out


def _dct_low(rgb_u8, k):
    """灰度图 2D-DCT 的低频 k×k 块（JPEG 式手工描述子）。"""
    out = np.empty((rgb_u8.shape[0], k * k), np.float32)
    for i in range(rgb_u8.shape[0]):
        g = cv2.cvtColor(rgb_u8[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        out[i] = cv2.dct(g)[:k, :k].reshape(-1)
    return out


def classical_desc(rgb_all, mask_all, name):
    """按 name 生成手工描述子 (N, D) float32。命名一律在 CLS_NAMES 里登记。"""
    if name[0] == "r":                                     # r{res}_{raw|norm|bgnorm|fgnorm}
        head, tail = name.split("_", 1)
        f = _res_u8(rgb_all, int(head[1:])).astype(np.float32)
        if tail in ("bgnorm", "fgnorm"):
            m = _mask_u8(mask_all, int(head[1:]))[..., None].astype(np.float32) / 255.0
            f = _std_all(f * ((1.0 - m) if tail == "bgnorm" else m))
        elif tail == "norm":
            f = _std_all(f)
        return f.reshape(f.shape[0], -1)
    if name.startswith("blur"):                            # blur{σ}_r{res}_{norm}
        sigma, rest = name[4:].split("_", 1)
        res, tail = int(rest.split("_")[0][1:]), rest.split("_", 1)[1]
        out = np.empty((rgb_all.shape[0], res, res, 3), np.uint8)
        for i in range(rgb_all.shape[0]):
            b = cv2.GaussianBlur(rgb_all[i], (0, 0), float(sigma))
            out[i] = cv2.resize(b, (res, res), interpolation=cv2.INTER_AREA)
        f = out.astype(np.float32)
        return (_std_all(f) if tail == "norm" else f).reshape(f.shape[0], -1)
    if name.startswith("gridms"):                          # gridms{g}_r{res}
        return _block_ms(_res_u8(rgb_all, int(name.split("_")[-1][1:])).astype(np.float32),
                         int(name[len("gridms")]))
    if name.startswith("pyr"):                             # pyr_8_16_32：多尺度低频谱直接拼接
        return np.concatenate(
            [_std_all(_res_u8(rgb_all, int(r)).astype(np.float32)).reshape(len(rgb_all), -1)
             for r in name.split("_")[1:]], axis=1)
    if name.startswith("huehsv"):                          # huehsv_r{res}
        return _hsv_hist(_res_u8(rgb_all, int(name.split("_")[-1][1:])))
    if name.startswith("dct"):                             # dct{k}_gray
        return _dct_low(_res_u8(rgb_all, WORK), int(name[3:].split("_")[0]))
    if name.startswith(("hsv_", "lab_", "labab_", "ycc_", "ycccbcr_")):
        space, rest = name.split("_", 1)
        chans = None
        if space == "labab":
            space, chans = "lab", [1, 2]
        elif space == "ycccbcr":
            space, chans = "ycc", [1, 2]
        res, tail = int(rest.split("_")[0][1:]), rest.split("_", 1)[1]
        f = _cvt_all(_res_u8(rgb_all, res), space).astype(np.float32)
        if chans is not None:
            f = f[..., chans]
        return (_std_all(f) if tail == "norm" else f).reshape(f.shape[0], -1)
    raise ValueError(f"未知描述子：{name}")


@torch.no_grad()
def sim(q_desc, db_desc, dev):
    """余弦相似度矩阵 (Q, DB)。"""
    q = F.normalize(torch.from_numpy(q_desc).to(dev).float(), dim=1)
    db = F.normalize(torch.from_numpy(db_desc).to(dev).float(), dim=1)
    return q @ db.T


def metrics(S, q_src, db_src, q_ident, db_ident):
    """由相似度矩阵算帧级/视频级 Top-K 命中率。"""
    res = {}
    for k in KS:
        idx = S.topk(k, dim=1).indices.cpu().numpy()
        res[f"frame_top{k}"] = float((db_src[idx] == q_src[:, None]).any(axis=1).mean())
        res[f"video_top{k}"] = float((db_ident[idx] == q_ident[:, None]).any(axis=1).mean())
    return res


def score(q_desc, db_desc, dev, q_src, db_src, q_ident, db_ident):
    """余弦检索（查询/图库分开传入）；返回帧级/视频级 Top-K 命中率。"""
    S = sim(q_desc, db_desc, dev)
    res = metrics(S, q_src, db_src, q_ident, db_ident)
    del S
    torch.cuda.empty_cache()
    return res


def main():
    cfg = load_config(CFG)
    cache = str(cfg.get("mask_cache_dir", "D:/py/AURA/baseline/cache/masks"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} | 掩膜缓存={cache}", flush=True)

    index = build_ffpp_index(cfg)
    samples = index["test"]
    gallery = [s for s in samples if s["kind"] == "real"]
    query = [s for s in samples if s["kind"] == "fake"]
    ng = len(gallery)
    print(f"[data] 图库(真实)={ng} 查询(伪造)={len(query)} 随机 Top-1={1/ng:.6f}", flush=True)

    rgb_all, mask_all = load_all([(s["path"], s) for s in gallery] + [(s["path"], s) for s in query],
                                 cache)

    q_src = np.array([s["source_id"] for s in query])
    q_ident = np.array([s["identity"] for s in query])
    db_src = np.array([s["id"] for s in gallery])
    db_ident = np.array([s["identity"] for s in gallery])

    results = {"n_gallery": ng, "n_query": len(query), "rows": []}
    print("\n" + "=" * 100)
    print(f"{'区域':<6}{'res':>5}{'mode':<8}{'帧Top-1':>11}{'帧Top-5':>11}{'视频Top-1':>12}{'视频Top-5':>12}")
    print("-" * 100)
    for region, res, mode in CONFIGS:
        d = descriptors(rgb_all, mask_all, region, res, mode)
        sc = score(d[ng:], d[:ng], dev, q_src, db_src, q_ident, db_ident)
        print(f"{region:<6}{res:>5}{mode:<8}{sc['frame_top1']*100:>10.2f}%{sc['frame_top5']*100:>10.2f}%"
              f"{sc['video_top1']*100:>11.2f}%{sc['video_top5']*100:>11.2f}%", flush=True)
        results["rows"].append({"region": region, "res": res, "mode": mode, **sc,
                                "dim": int(d.shape[1])})
        del d
    print("=" * 100)
    print("参照：零训练 tiny32 整图 = 58.80%（帧Top-1）/ 87.15%（帧Top-5）/ 99.75%（视频Top-1）；"
          "自建五轮最好 47.75%")

    # ---------- 手工描述子扫描：传统 CV 能否超过 58.71%？----------
    print("\n" + "=" * 100)
    print("手工描述子扫描（零训练）：58.71% 是否为协议天花板？")
    print(f"{'描述子':<22}{'维度':>9}{'帧Top-1':>11}{'帧Top-5':>11}{'视频Top-1':>12}")
    print("-" * 100)
    cls_rows = []
    keep = {}
    cls_best = ("", -1.0)
    for name in CLS_NAMES:
        d = classical_desc(rgb_all, mask_all, name)
        sc = score(d[ng:], d[:ng], dev, q_src, db_src, q_ident, db_ident)
        print(f"{name:<22}{d.shape[1]:>9}{sc['frame_top1']*100:>10.2f}%{sc['frame_top5']*100:>10.2f}%"
              f"{sc['video_top1']*100:>11.2f}%", flush=True)
        cls_rows.append({"name": name, "dim": int(d.shape[1]), **sc})
        if sc["frame_top1"] > cls_best[1]:
            cls_best = (name, sc["frame_top1"])
        if name in FUSE_NAMES:
            keep[name] = d                                     # 供晚融合复用
        else:
            del d                                              # 及时释放（单条最大 118MB+）
    print(f"  单体最佳：{cls_best[0]} = {cls_best[1]*100:.2f}%", flush=True)

    # 晚融合：两个相似度矩阵加权后重排（α 网格搜索）
    print("-" * 100)
    print("晚融合 S = S_a + α·S_b（各分路单独都低于整图，若误差不重叠则可叠加）：")
    print(f"{'组合':<32}{'α':>6}{'帧Top-1':>11}{'帧Top-5':>11}")
    print("-" * 100)
    for a, b in FUSE_PAIRS:
        Sa = sim(keep[a][ng:], keep[a][:ng], dev)
        Sb = sim(keep[b][ng:], keep[b][:ng], dev)
        for alpha in FUSE_ALPHAS:
            sc = metrics(Sa + alpha * Sb, q_src, db_src, q_ident, db_ident)
            tag = f"{a}+{alpha:g}*{b}"
            print(f"{tag:<32}{alpha:>6g}{sc['frame_top1']*100:>10.2f}%{sc['frame_top5']*100:>10.2f}%",
                  flush=True)
            cls_rows.append({"name": tag, "dim": -1, **sc})
            if sc["frame_top1"] > cls_best[1]:
                cls_best = (tag, sc["frame_top1"])
        del Sa, Sb
        torch.cuda.empty_cache()
    print(f"  含融合后最佳：{cls_best[0]} = {cls_best[1]*100:.2f}%", flush=True)

    # PCA 白化：在图库上拟合（检索文献里最经典的零训练增益）
    print("-" * 100)
    print("PCA 白化（在图库 1600 张上拟合），输入 r32_norm：")
    d32 = keep["r32_norm"]
    mu_db = d32[:ng].mean(axis=0, keepdims=True)
    X = d32[:ng] - mu_db
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    for k in (128, 256, 512):
        W = vt[:k].T / (sv[:k] + 1e-6)
        sc = score((d32[ng:] - mu_db) @ W, X @ W, dev, q_src, db_src, q_ident, db_ident)
        print(f"  r32_norm + PCA白化 k={k:<5} 帧Top-1={sc['frame_top1']*100:6.2f}% "
              f"帧Top-5={sc['frame_top5']*100:6.2f}%", flush=True)
        cls_rows.append({"name": f"r32_norm_pca{k}", "dim": k, **sc})
        if sc["frame_top1"] > cls_best[1]:
            cls_best = (f"r32_norm_pca{k}", sc["frame_top1"])
    print("=" * 100)
    print(f"判读：若全部 ≤ 58.71%，协议饱和坐实；最佳 {cls_best[0]} = {cls_best[1]*100:.2f}%", flush=True)
    results["classical"] = cls_rows
    results["cls_best"] = {"name": cls_best[0], "frame_top1": cls_best[1]}

    # ---------- 退化对照：图库干净、查询逐档退化（与模型 §3.8 完全同协议）----------
    # 模型 E 轮 e6 余弦帧级 Top-1：clean 46.44 / jpeg50 45.25 / jpeg20 42.52 /
    #                              scale0.5 46.20 / crop0.5 0.33 / reencode 44.74
    print("\n" + "=" * 100)
    print("退化对照（零训练描述子）：图库保持干净、查询逐档退化 —— 与模型同协议")
    print(f"{'描述子':<18}{'档位':<12}{'帧Top-1':>11}{'帧Top-5':>11}{'视频Top-1':>12}{'视频Top-5':>12}")
    print("-" * 100)
    g_rgb, g_mask = rgb_all[:ng], mask_all[:ng]
    q_rgb_clean, q_mask = rgb_all[ng:], mask_all[ng:]
    deg_rows = []
    for region, res, mode in DEG_DESCS:
        tag = f"{region}@{res}:{mode}"
        dbd = descriptors(g_rgb, g_mask, region, res, mode)          # 干净图库，只编码一次
        for level in DEG_LEVELS:
            q_rgb = q_rgb_clean if level == "clean" else np.stack(
                [apply_degradation(x, level) for x in q_rgb_clean])
            qd = descriptors(q_rgb, q_mask, region, res, mode)
            sc = score(qd, dbd, dev, q_src, db_src, q_ident, db_ident)
            print(f"{tag:<18}{level:<12}{sc['frame_top1']*100:>10.2f}%{sc['frame_top5']*100:>10.2f}%"
                  f"{sc['video_top1']*100:>11.2f}%{sc['video_top5']*100:>11.2f}%", flush=True)
            deg_rows.append({"desc": tag, "level": level, **sc})
            del qd
        del dbd
        torch.cuda.empty_cache()
    print("=" * 100)
    print("对照：模型 E 轮 e6 余弦帧级 Top-1 = clean 46.44 / jpeg50 45.25 / jpeg20 42.52 / "
          "scale0.5 46.20 / crop0.5 0.33 / reencode 44.74")
    results["degradation"] = deg_rows

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
