"""全图对比学习的数据管线：退化增强 + 人脸掩膜擦除 + 同源分组采样。

与 baseline 的差异：不做前景/背景分离，整图直接喂编码器。正样本来自"同一个源视频 X"
（同一段真实视频的不同帧、不同换脸算法 / 不同供体，以及各自的退化版本），
负样本来自"不同源视频"。这正对应用户的要求："同源不同攻击、退化图片靠的近，不同源离得远"。

掩膜擦除（本轮新增，训练视图按 erase_prob 概率施加）：FF++ 伪造帧命名 X_Y 里的 X 就是
检索答案（背景来源视频），而人脸区域会连同 X 的头姿、光照、边界和整体色调映射一起被换脸带
进来 —— 模型能直接读人脸区域"蒙对"视频级答案，不必真读懂背景。实测正对上这一点：视频级
Top-1 已 99.48%，帧级却卡在 47.75%（还不如零训练 32×32 缩略图的 58.71%）。把人脸区域
随机填充掉，就是掐断这条捷径，逼模型只能看背景。
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# 训练视图的随机退化范围。当前评测协议下查询帧与库中源帧是【同序号帧】（近乎逐像素相同），
# 不加退化的话模型抄近路记住像素即可，学不到可迁移的"来源"表征 —— 退化正是逼它看背景/场景。
_JPEG_QUALITY = (30, 95)
_DOWNSCALE = (0.4, 1.0)

# 退化鲁棒基准（冻结六档，见 docs 的 §5 评测协议）。只作用于【查询】，图库保持干净：
# 真实取证场景里图库是干净的归档素材，而待查的伪造图经过社媒转存、二次压缩、裁剪等。
DEGRADATION_LEVELS = ("clean", "jpeg50", "jpeg20", "scale0.5", "crop0.5", "reencode")

# 几何扰动档（方案 B：更难的协议）。动机：现有六档只动高频（JPEG/缩放/重编码），而任务判别信号
# 恰在低频全局色调里 —— 零训练 32x32 只保低频，故对那六档天然不变（实测 58.71% -> 58.80%）。
# 要拉开"训练 vs 零训练"的差距，必须动【几何】：全局平移/旋转/缩放/非中心裁剪会让原始像素
# 描述子整体错位、直接崩掉；而深度编码器 head 前的 GAP 对全局平移近似不变，有机会守住。
# 依旧只作用于【查询】，图库保持干净。全部确定性（评测可复现）。
GEOM_LEVELS = ("crop0.8", "crop0.7off", "zoom1.15", "rot5", "rot10", "shift8")


def apply_degradation(img_rgb: np.ndarray, level: str) -> np.ndarray:
    """按档位名做【确定性】退化（评测用，不含随机）。六档口径见 DEGRADATION_LEVELS。"""
    if level in (None, "", "clean"):
        return img_rgb
    h, w = img_rgb.shape[:2]
    if level.startswith("jpeg"):
        q = int(level[4:])
        ok, buf = cv2.imencode(".jpg", img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img_rgb
    if level == "scale0.5":                                  # 0.5× 缩放（下采样再插回）：丢高频细节
        small = cv2.resize(img_rgb, (max(8, w // 2), max(8, h // 2)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if level == "crop0.5":                                   # 中心裁剪 50% 再插回：丢外围背景
        ch, cw = h // 2, w // 2
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return cv2.resize(img_rgb[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    if level == "reencode":                                  # 社媒重编码链：0.75× -> JPEG70 -> 插回 -> JPEG60
        small = cv2.resize(img_rgb, (max(8, int(w * 0.75)), max(8, int(h * 0.75))),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            small = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img_rgb = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        ok, buf = cv2.imencode(".jpg", img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img_rgb
    if level in ("crop0.8", "crop0.7off", "zoom1.15"):       # 裁剪/放大类：让背景整体位移或缩放
        if level == "crop0.8":
            ch, cw = int(round(h * 0.8)), int(round(w * 0.8))
            y0, x0 = (h - ch) // 2, (w - cw) // 2
        elif level == "crop0.7off":                          # 非中心裁剪：位移（方向/偏移固定，可复现）
            ch, cw = int(round(h * 0.7)), int(round(w * 0.7))
            y0, x0 = int(round(h * 0.10)), int(round(w * 0.05))
        else:                                                # 放大 15%：取 1/1.15 中心区再插回
            ch, cw = int(round(h / 1.15)), int(round(w / 1.15))
            y0, x0 = (h - ch) // 2, (w - cw) // 2
        return cv2.resize(img_rgb[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    if level.startswith("rot"):                              # 绕中心旋转 +N 度，边缘复制填充
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(level[3:]), 1.0)
        return cv2.warpAffine(img_rgb, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    if level.startswith("shift"):                            # 固定方向平移 N 像素，边缘复制填充
        d = int(level[5:])
        m = np.float32([[1, 0, d], [0, 1, d * 0.75]])
        return cv2.warpAffine(img_rgb, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    raise ValueError(f"未知退化档位: {level}")


def _geom_aug(img_rgb: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """随机几何增强（方案 B / F 轮新增）：旋转 ±10°、缩放 0.7~1.15、平移 ±8px。

    动机：现有增强只有翻转/降采样/JPEG/模糊/噪声/亮度，全是【光度】退化；而"更难协议"要的是
    让模型对【几何】变化不变。实测（零训练 tiny32 vs E 轮模型，见 geom_model_e6.json）：
    tiny32 在 rot5 掉到 41.0%、rot10 25.9%、shift8 30.8%、zoom1.15 25.7%、crop0.8 10.5%，
    而从未见过几何增强的模型在 rot10 已反超（30.53% vs 25.94%）—— 说明 conv+GAP 自带部分
    几何不变性，加入几何增强应能把 rot/shift/zoom 档进一步拉开。
    图与掩膜必须共用同一仿射矩阵（否则擦除会错位到背景上）。
    """
    h, w = img_rgb.shape[:2]
    ang = float(rng.uniform(-10.0, 10.0)) if rng.random() < 0.7 else 0.0
    sc = float(rng.uniform(0.7, 1.15)) if rng.random() < 0.6 else 1.0
    dx = float(rng.integers(-8, 9)) if rng.random() < 0.6 else 0.0
    dy = float(rng.integers(-8, 9)) if rng.random() < 0.6 else 0.0
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, sc)
    m[0, 2] += dx
    m[1, 2] += dy
    img_rgb = cv2.warpAffine(img_rgb, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    if mask is not None:
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return img_rgb, mask


def degrade(img_rgb: np.ndarray, rng: np.random.Generator,
            mask: np.ndarray = None, geom: bool = False) -> np.ndarray:
    """对 uint8 RGB 图做随机退化：水平翻转 / 几何（可选）/ 分辨率下采样 / JPEG / 模糊 / 噪声 / 亮度。

    mask 非 None 时，最后再把人脸区域擦掉（见 erase_face）；掩膜会跟着翻转与几何变换同步走。
    geom=True 时追加随机旋转/缩放/平移（F 轮起启用，见 _geom_aug）。
    """
    h, w = img_rgb.shape[:2]
    if rng.random() < 0.5:                                   # 水平翻转
        img_rgb = np.ascontiguousarray(img_rgb[:, ::-1])
        if mask is not None:                                 # 掩膜必须同步翻转，否则擦到背景上
            mask = np.ascontiguousarray(mask[:, ::-1])
    if geom:                                                 # 几何增强（旋转/缩放/平移）
        img_rgb, mask = _geom_aug(img_rgb, mask, rng)
    if rng.random() < 0.6:                                   # 分辨率退化（下采样再插回）
        s = float(rng.uniform(*_DOWNSCALE))
        small = cv2.resize(img_rgb, (max(8, int(w * s)), max(8, int(h * s))),
                           interpolation=cv2.INTER_AREA)
        interp = cv2.INTER_LINEAR if rng.random() < 0.8 else cv2.INTER_NEAREST
        img_rgb = cv2.resize(small, (w, h), interpolation=interp)
    if rng.random() < 0.6:                                   # JPEG 重压缩
        q = int(rng.integers(*_JPEG_QUALITY))
        ok, buf = cv2.imencode(".jpg", img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img_rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if rng.random() < 0.3:                                   # 高斯模糊
        k = int(rng.choice([3, 5, 7]))
        img_rgb = cv2.GaussianBlur(img_rgb, (k, k), 0)
    if rng.random() < 0.3:                                   # 高斯噪声
        noise = rng.normal(0.0, float(rng.uniform(2.0, 10.0)), img_rgb.shape)
        img_rgb = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:                                   # 亮度 / 对比度抖动
        alpha, beta = float(rng.uniform(0.8, 1.2)), float(rng.uniform(-20, 20))
        img_rgb = np.clip(img_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if mask is not None:                                     # 最后擦人脸：填充物不再被上面几步"抹平"
        img_rgb = erase_face(img_rgb, mask, rng)
    return np.ascontiguousarray(img_rgb)


# ---------- 人脸区域擦除 ----------
# 动机：FF++ 伪造帧 X_Y 的检索答案就是 original/X 的【同序号帧】（见 baseline/data/dataset.py），
# 背景逐像素相同；而人脸区域会连同 X 的头姿、光照、换脸边界和整体色调映射一起被带进来。模型
# 因此可以只读人脸区域就"蒙对"视频级答案，不必真的读懂背景 —— 实测正对上这一点：视频级 Top-1
# 已 99.48%，帧级却卡在 47.75%（还不如零训练 32×32 缩略图的 58.71%）。训练视图按 erase_prob
# 概率把人脸区域随机填充掉，就是掐断这条捷径。
_MASK_SIZE = 128                     # BiSeNet 掩膜磁盘缓存的解析尺寸（= baseline 的 image_size）
_FILL_MODES = ("noise", "solid", "hole", "blob")
_provider = None


def _mask_provider(cache_dir: str):
    """惰性建一个 FaceParseMask 只为复用它的缓存键算法；这里绝不触发解释网络。"""
    global _provider
    if _provider is None:
        from baseline.data.preprocessing import FaceParseMask
        _provider = FaceParseMask(_MASK_SIZE, cache_dir=os.path.abspath(cache_dir))
    return _provider


def load_face_mask(path: str, size: int, cache_dir: str):
    """取该图的人脸前景软掩膜并缩放到 (size, size)；缓存未命中返回 None（跳过擦除）。

    只读 baseline 预计算好的 .npy 缓存（49288 张 128×128 uint8）；刻意不走 mask_for_path ——
    缓存未命中时它会去加载 BiSeNet，既慢又要抢显存。
    """
    prov = _mask_provider(cache_dir)
    f = os.path.join(os.path.abspath(cache_dir),
                     prov._cache_key(os.path.abspath(path), _MASK_SIZE) + ".npy")
    if not os.path.exists(f):
        return None
    m = np.load(f)                                                  # uint8 (128,128)，0~255
    if m.shape != (_MASK_SIZE, _MASK_SIZE):
        return None
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR)  # 放大到模型输入，边缘自带羽化
    return m.astype(np.float32) / 255.0


def erase_face(img_rgb: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """把人脸区域替换成随机填充：噪点 / 纯色 / 空洞 / 随机形状。

    阈值、填充方式、形状位置与颜色全部逐样本随机 —— 否则"填充物本身"会变成一条新的可学线索，
    模型只要认出"这块是被擦的"就能知道答案边界，等于换个捷径继续抄。
    """
    h, w = img_rgb.shape[:2]
    # 阈值只在 (0.35, 0.65) 里随机：实测 BiSeNet 缓存的掩膜近似【二值】（阈值抬到 0.6/0.7 时擦除
    # 面积几乎不变：0.556 -> 0.552 -> 0.551），所以"收紧阈值"并不能减小擦除范围 —— 控制擦除强度的
    # 唯一有效旋钮是 erase_prob（见 ContrastiveDataset）。随机化阈值只用于打散掩膜边缘。
    a = (mask > float(rng.uniform(0.35, 0.65))).astype(np.float32)   # 软掩膜 -> 随机阈值二值化
    if a.mean() < 0.01:                                             # 掩膜太小/解析失败 -> 不擦
        return img_rgb
    mode = str(rng.choice(_FILL_MODES))
    if mode == "noise":                                             # 噪点
        fill = rng.integers(0, 256, img_rgb.shape, dtype=np.uint8)
    elif mode == "solid":                                           # 纯色
        fill = np.broadcast_to(rng.integers(0, 256, 3, dtype=np.uint8), img_rgb.shape).copy()
    elif mode == "hole":                                            # 空洞（纯黑）
        fill = np.zeros_like(img_rgb)
    else:                                                           # 随机形状：底色 + 若干随机椭圆
        fill = np.broadcast_to(rng.integers(0, 256, 3, dtype=np.uint8), img_rgb.shape).copy()
        ys, xs = np.nonzero(a)
        for _ in range(int(rng.integers(1, 4))):
            cx = int(rng.integers(xs.min(), xs.max() + 1))
            cy = int(rng.integers(ys.min(), ys.max() + 1))
            rx = int(rng.integers(max(4, w // 16), max(8, w // 4)))
            ry = int(rng.integers(max(4, h // 16), max(8, h // 4)))
            col = tuple(int(v) for v in rng.integers(0, 256, 3))
            cv2.ellipse(fill, (cx, cy), (rx, ry), int(rng.integers(0, 180)), 0, 360, col, -1)
    a3 = a[:, :, None]
    out = img_rgb.astype(np.float32) * (1.0 - a3) + fill.astype(np.float32) * a3
    return np.clip(out, 0, 255).astype(np.uint8)


class ContrastiveDataset(Dataset):
    """把一个 split 的样本（真实帧 + 伪造帧）读成整图张量。

    返回 dict：image (3,S,S) float[0,1]、label（源视频 X 的整数标签）、
    id / identity / source_id（评估用）。
    """

    def __init__(self, samples, image_size: int = 256, train: bool = True, seed: int = 42,
                 erase_prob: float = 0.0, mask_cache_dir: str = "cache/masks",
                 degrade_level: str = None, label_key: str = "identity",
                 geom_aug: bool = False):
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.rng = np.random.default_rng(seed)
        self.erase_prob = float(erase_prob)
        self.mask_cache_dir = str(mask_cache_dir)
        # 几何增强（F 轮起启用）：训练视图追加随机旋转/缩放/平移，见 _geom_aug。
        # 目的：让模型对"更难的几何协议"（rot5/rot10/shift8/zoom1.15）保持稳定。
        self.geom_aug = bool(geom_aug)
        # 退化鲁棒基准：非 None 时对每张图施加【确定性】退化（只用于评估视图，见 apply_degradation）
        self.degrade_level = degrade_level
        # 对比学习的类别粒度，可配置：
        #   "identity"  = 源视频 X（粗粒度，同视频 8 帧互为【正样本】）
        #   "source_id" = 具体源帧 X_f{帧号}（细粒度，与帧级判据对齐；同视频其他帧变【负样本】）
        # 已测：粗粒度 6 轮帧级 47.75% / 视频级 99.48%；细粒度的那一轮只有 38.92% / 91.16%，
        # 但那一轮训练极不稳（e1 3.56% 后反复跳变），疑似"同视频相邻帧"这种近乎相同的极难负样本
        # 所致，故本轮把 lr / warmup 一并调稳后重测，不沿用旧结论。见 docs/AURA_HANDOFF.md §6。
        self.label_key = str(label_key)
        keys = sorted({s[self.label_key] for s in self.samples})
        self.id2lab = {k: i for i, k in enumerate(keys)}
        self.labels = np.array([self.id2lab[s[self.label_key]] for s in self.samples], dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["path"], cv2.IMREAD_COLOR)       # BGR
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {s['path']}")
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.degrade_level:                              # 退化鲁棒基准：对评估视图做确定性退化
            img = apply_degradation(img, self.degrade_level)
        if self.train:
            mask = None
            if self.erase_prob > 0 and self.rng.random() < self.erase_prob:
                mask = load_face_mask(s["path"], self.image_size, self.mask_cache_dir)
            img = degrade(img, self.rng, mask, geom=self.geom_aug)   # 训练视图才加退化；评估视图保持原图
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().div_(255.0)
        return {"image": x, "label": int(self.labels[idx]), "id": s["id"],
                "identity": s["identity"], "source_id": s["source_id"]}


class GroupBatchSampler:
    """同源分组批采样：每批取 groups_per_batch 个源视频，每个源取 samples_per_group 张图。

    必须成组采样：均匀随机抽样下一个 batch 里几乎不可能出现两张同源图，监督对比损失的
    正样本项就恒为空。分组后每批既有正样本，也有足够多不同源可当负样本。
    """

    def __init__(self, labels, groups_per_batch: int, samples_per_group: int,
                 steps_per_epoch: int, seed: int = 42):
        self.groups_per_batch = int(groups_per_batch)
        self.samples_per_group = int(samples_per_group)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        groups = {}
        for i, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(i)
        self.groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
        self.keys = np.asarray(sorted(self.groups.keys()), dtype=np.int64)
        self.n_groups = len(self.keys)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 100003)
        g = min(self.groups_per_batch, self.n_groups)
        for _ in range(self.steps_per_epoch):
            picked = rng.choice(self.keys, size=g, replace=False)     # 本批的源视频
            idx = []
            for k in picked:
                pool = self.groups[int(k)]
                # 该源样本数不足时允许重复取，保证每批正样本数量稳定
                idx.extend(rng.choice(pool, size=self.samples_per_group,
                                      replace=len(pool) < self.samples_per_group).tolist())
            yield idx
