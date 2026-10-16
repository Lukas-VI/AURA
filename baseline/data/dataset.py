"""AURA 数据集加载（FF++ 抽帧版）。

数据来源：FF++ C23 抽帧结果（默认 D:/ffpp_frames），目录形如

    original/000/f000.jpg ...              # 1000 个真实视频，每个 8 帧（256x256 人脸裁剪）
    Deepfakes/000_003/f000.jpg ...         # 5 种换脸算法的伪造帧
    Face2Face/000_003/f000.jpg ...
    manifest.csv                           # 抽帧清单：folder,stem,frame,frame_idx,video,...,image

命名约定（FF++ 官方）：伪造视频 `X_Y.mp4` 中
    X = 目标视频：提供背景 / 头姿 / 动作（即“被换脸者”所在的那段真实视频）
    Y = 供体视频：提供人脸身份
因此对一张伪造帧 X_Y：
    * 它的【背景来源真实帧】 = original/X 的同序号帧  -> 检索答案（source_id 指向它）
    * 它的【人脸来源真实帧】 = original/Y 的同序号帧  -> 供体（donor），用作四元组负样本

数据划分：按【真实视频编号 X】切分 train/val/test。同一视频的帧必然落在同一个 split，
且伪造图与其源真实图也在同一 split —— 检索评估不会因为划分而泄漏答案。

说明：DeepFakeDetection 文件夹命名不同（形如 `01_02__meeting_serious__YVGY8LOK`），
它对应的 original 不属于本仓库这套 000..999（无法与 original/X 配对），故默认不纳入。
"""
import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils import ensure_dir
from .fakegen import SyntheticFaceSwap
from .preprocessing import FaceBackgroundSplit, FaceParseMask
from .transforms import build_transform


def load_manifest(data_root: str, manifest: str = "manifest.csv"):
    """读取抽帧清单，返回记录列表（每行一个 dict）。"""
    path = manifest if os.path.isabs(manifest) else os.path.join(data_root, manifest)
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _frame_path(data_root: str, folder: str, image_rel: str) -> str:
    """manifest 里的相对路径 -> 绝对路径（兼容 Windows 下 '/' 与 '\\' 混用）。"""
    return os.path.normpath(os.path.join(data_root, folder, image_rel))


def build_ffpp_index(cfg: dict):
    """扫描 manifest，按视频编号切分 train/val/test，返回 {split: [样本 dict, ...]}。

    样本 dict 字段
    --------------
    kind       : 'real' / 'fake'
    label      : 0=真实 / 1=伪造
    id         : 全局唯一 id（真实图是检索数据库里的 key）
    identity   : 背景来源视频编号 X（划分与分组都按它）
    source_id  : 检索答案 —— 该样本对应的真实源帧 id（真实图就是它自己）
    path       : 本样本自己的图片路径（伪造图就是伪造帧本身）
    source_path: 背景来源真实帧路径（伪造图用；供阶段2 取查询 q）
    donor_path : 人脸来源真实帧路径（伪造图用；供阶段2 取负样本 q−）
    """
    root = cfg["data_root"]
    real_folder = str(cfg.get("real_folder", "original"))
    manipulations = list(cfg.get("manipulations", []))
    n_frames = int(cfg.get("frames_per_video", 1))

    records = load_manifest(root, cfg.get("manifest", "manifest.csv"))

    # ---------- 1. 整理 manifest：真实视频 -> 帧；伪造视频 -> 帧 ----------
    real_frames = {}      # {视频号 X: [(帧序号, 路径), ...]}
    fake_map = {}         # {(X, 文件夹, stem): {"y": 供体号Y, "frames": [(帧序号, 路径), ...]}}
    for r in records:
        folder, stem, frame = r["folder"], r["stem"], int(r["frame"])
        p = _frame_path(root, folder, r["image"])
        if folder == real_folder:
            real_frames.setdefault(stem, []).append((frame, p))
        elif folder in manipulations:
            parts = stem.split("_")
            if len(parts) < 2:                        # 命名不符合 X_Y（如 DeepFakeDetection），跳过
                continue
            x, y = parts[0], parts[1]
            entry = fake_map.setdefault((x, folder, stem), {"y": y, "frames": []})
            entry["frames"].append((frame, p))
    for flist in real_frames.values():
        flist.sort(key=lambda t: t[0])                # 按帧序号排序，保证取前 N 帧稳定

    # 伪造视频按“背景来源视频 X”归组，便于按 split 取用
    fake_by_x = {}        # {视频号 X: [(伪造文件夹, stem, 供体号 Y, [(帧序号,路径),...]), ...]}
    for (x, folder, stem), info in sorted(fake_map.items()):
        frames = sorted(info["frames"], key=lambda t: t[0])
        fake_by_x.setdefault(x, []).append((folder, stem, info["y"], frames))

    # 供体帧的全量查表（供体视频可能不在当前 split，但取它的真实图作负样本是允许的）
    real_by_frame = {x: {fr: p for fr, p in flist} for x, flist in real_frames.items()}

    # ---------- 2. 按视频编号切分 ----------
    videos = sorted(real_frames.keys())
    n_tr, n_va, n_te = (int(cfg["num_train_videos"]), int(cfg["num_val_videos"]),
                        int(cfg["num_test_videos"]))
    if n_tr + n_va + n_te > len(videos):
        raise ValueError(f"视频总数 {len(videos)} 不足以切分 {n_tr}+{n_va}+{n_te}")
    split_videos = {
        "train": videos[:n_tr],
        "val": videos[n_tr:n_tr + n_va],
        "test": videos[n_tr + n_va:n_tr + n_va + n_te],
    }

    # ---------- 3. 生成样本 ----------
    index = {}
    for split, vids in split_videos.items():
        samples = []
        n_real = n_fake = 0
        for x in vids:
            frames = real_frames[x][:n_frames]            # 本 split 内该视频取用的帧
            frame_map = dict(frames)                      # 帧序号 -> 真实帧路径（检索答案表）
            for fr, p in frames:                          # ---- 真实样本 ----
                rid = f"{x}_f{fr:03d}"
                samples.append({"kind": "real", "label": 0, "id": rid, "identity": x,
                                "source_id": rid, "path": p, "source_path": p, "donor_path": None})
                n_real += 1
            for folder, stem, y, flist in fake_by_x.get(x, []):   # ---- 伪造样本 ----
                donor_frames = real_by_frame.get(y)
                if not donor_frames:                      # 供体视频缺失 -> 跳过（无法构造负样本）
                    continue
                donor_default = donor_frames[min(donor_frames)]   # 供体第一帧（兜底）
                for fr, fp in flist[:n_frames]:
                    src_path = frame_map.get(fr)          # 背景来源：同序号真实帧
                    if src_path is None:                  # 该帧不参与本 split -> 跳过
                        continue
                    samples.append({
                        "kind": "fake", "label": 1, "id": f"{folder}_{stem}_f{fr:03d}",
                        "identity": x, "source_id": f"{x}_f{fr:03d}",
                        "path": fp, "source_path": src_path,
                        "donor_path": donor_frames.get(fr, donor_default),
                    })
                    n_fake += 1
        index[split] = samples
        print(f"[data] {split}: {len(vids)} 个视频 -> {n_real} 张真实 + {n_fake} 张伪造"
              f"（{len(manipulations)} 种篡改算法）")
    return index


class AURADataset(Dataset):
    """AURA 基线数据集。

    参数
    ----
    samples : list[dict]   build_ffpp_index 产出的某一切分样本列表
    split : str            'train' / 'val' / 'test'（train 启用随机水平翻转）
    image_size : int       缩放后的模型输入边长
    transform : 可选，覆盖默认 transform
    mask_source : str      'parse'（BiSeNet 人脸解析，默认）/ 'geometry'（居中椭圆，消融）
    mask_cache_dir : str   解析掩膜的磁盘缓存目录（仅 mask_source='parse' 时使用）
    """

    def __init__(self, samples, split, image_size: int = 128, transform=None, seed: int = 42,
                 mask_source: str = "parse", mask_cache_dir: str = None,
                 center_ratio=(0.50, 0.62), expand: float = 1.0):
        self.samples = list(samples)
        self.split = split
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)
        # 随机翻转已下移到读图阶段（与掩膜同步），故这里只有确定性的张量化
        self.transform = transform or build_transform()
        # 前后景掩膜提供器：'parse' 用 BiSeNet 真实人脸解析；'geometry' 用居中椭圆先验
        self.splitter = self._build_splitter(mask_source, mask_cache_dir, center_ratio, expand)
        # 合成换脸器：诊断实验、阶段2 构造 q+1 时复用（与掩膜提供器共用同一套前景定义）
        self.faker = SyntheticFaceSwap(self.splitter)

    def _build_splitter(self, mask_source, cache_dir, center_ratio, expand):
        """掩膜提供器：'parse' -> BiSeNet 人脸解析（带缓存）；其余 -> 居中椭圆几何先验。"""
        if str(mask_source).lower() == "parse":
            return FaceParseMask(self.image_size, center_ratio=center_ratio,
                                 expand=expand, cache_dir=cache_dir)
        return FaceBackgroundSplit(self.image_size, expand=expand, center_ratio=center_ratio)

    def __len__(self):
        return len(self.samples)

    # ---------- 图像读取 ----------
    def _read(self, path):
        """读图，返回原始分辨率的 BGR ndarray。"""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        return img

    def to_tensor(self, img_bgr):
        """BGR ndarray -> 缩放到 image_size -> RGB -> transform -> tensor。

        必须走这里：FF++ 抽帧出来是 256x256，模型输入是 128，直接喂原图会显著增大显存。
        """
        img = cv2.resize(img_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # cv2 是 BGR，torchvision 期望 RGB
        return self.transform(img)

    def _load_mask(self, path, img_bgr=None):
        """取该图 (image_size, image_size) 的前景掩膜（float32，np.array）。

        'parse' 走磁盘缓存（不需要 img_bgr）；'geometry' 需要原图，可由调用方传入避免重复读盘。
        """
        prov = self.splitter
        if isinstance(prov, FaceParseMask):
            return prov.mask_for_path(path, self.image_size)
        img = img_bgr if img_bgr is not None else self._read(path)
        m = prov.mask(img)
        return cv2.resize(m, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    def warm_mask_cache(self, verbose: bool = True) -> int:
        """预热本数据集全部图片（含源图/供体图）的解析掩膜缓存，并释放解析网络。

        必须在建模型之前调用：解析网络与 HashVAE 同时驻留 4GB 显存会有 OOM 风险。
        """
        prov = self.splitter
        if not isinstance(prov, FaceParseMask):
            return 0
        paths = {s[k] for s in self.samples for k in ("path", "source_path", "donor_path") if s.get(k)}
        n = prov.prebuild(paths, verbose=verbose)
        prov.unload()
        return n

    # ---------- 主入口 ----------
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self._read(s["path"])                    # 伪造样本读的就是伪造帧本身
        mask = self._load_mask(s["path"], img)         # 前景掩膜（与图同源）
        if self.split == "train" and self.rng.random() < 0.5:   # 训练随机翻转：图与掩膜必须同步
            img = cv2.flip(img, 1)
            mask = np.ascontiguousarray(np.fliplr(mask))
        return {
            "image": self.to_tensor(img),
            "mask": torch.from_numpy(mask)[None],      # (1,S,S) 前景掩膜，随样本送入模型
            "label": torch.tensor(s["label"], dtype=torch.float32),
            "id": s["id"],
            "identity": s["identity"],
            "source_id": s["source_id"],
            "kind": s["kind"],
            "path": s["path"],
        }


class QuadrupletDataset(Dataset):
    """阶段 2 用的四元组数据集（逐个真实伪造图取样）。

    每个样本给出一组三张图：
        img_q     —— 该伪造图的【背景来源真实帧】（original/X 同序号帧），即查询 q；
        img_fake  —— 真实伪造帧本身（背景来自 q、人脸来自供体视频 Y），即 q+2；
        img_donor —— 供体视频 Y 的真实帧，作为四元组负样本 q−（不同背景 + 不同身份）。
    前景正样本 q+1（保留 q 的人脸、换到供体背景）在训练脚本里用合成换脸实时构造。
    """

    def __init__(self, aura_dataset, indices=None, seed: int = 42):
        self.ds = aura_dataset
        self.indices = list(indices) if indices is not None else [
            i for i, s in enumerate(aura_dataset.samples) if s["kind"] == "fake"
        ]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s = self.ds.samples[self.indices[i]]
        paths = [s["source_path"],   # q
                 s["path"],          # 真实伪造图 q+2
                 s["donor_path"]]    # 负样本 q−
        imgs = [self.ds._read(p) for p in paths]
        masks = [self.ds._load_mask(p, im) for p, im in zip(paths, imgs)]   # 三张图各自的前景掩膜
        # 三张图共享同一次随机翻转（掩膜同步翻转）：否则背景/人脸配对会被打乱，四元组就失效了
        if self.ds.split == "train" and self.rng.random() < 0.5:
            imgs = [cv2.flip(x, 1) for x in imgs]
            masks = [np.ascontiguousarray(np.fliplr(m)) for m in masks]
        q, f, d = (self.ds.to_tensor(x) for x in imgs)
        mq, mf, md = (torch.from_numpy(m)[None] for m in masks)             # 各 (1,S,S)
        return {"img_q": q, "img_fake": f, "img_donor": d,
                "mask_q": mq, "mask_fake": mf, "mask_donor": md}


def build_datasets(cfg: dict, index=None):
    """根据配置构建 train/val/test 三个数据集，返回 (datasets, index)。"""
    index = index or build_ffpp_index(cfg)
    size = int(cfg["image_size"])
    seed = int(cfg.get("seed", 42))
    mask_source = str(cfg.get("fg_mask_source", "parse"))
    # 解析掩膜的磁盘缓存目录（相对 baseline 包目录，如 'cache/masks'）
    cache_dir = ensure_dir(cfg.get("mask_cache_dir", "cache/masks")) \
        if mask_source.lower() == "parse" else None
    datasets = {split: AURADataset(
        samples, split, image_size=size, seed=seed,
        mask_source=mask_source, mask_cache_dir=cache_dir,
        center_ratio=tuple(cfg.get("fg_center_ratio", [0.50, 0.62])),
        expand=float(cfg.get("fg_expand", 1.0)))
        for split, samples in index.items()}
    return datasets, index
