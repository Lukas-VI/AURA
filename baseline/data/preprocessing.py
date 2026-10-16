"""前景（人脸）/背景（非人脸）分离。

论文原本要求 RetinaFace + Dlib 68 点凸包。当前环境未安装 dlib / retinaface，且 OpenCV 5.0
已移除 Haar 级联。因此这里提供两种掩膜来源，二者实现同一接口 mask(img_bgr) -> float32(H,W)：

    1) FaceParseMask（默认，'parse'）——
       复用 third_party/SimSwap 里的 BiSeNet 人脸解析网络（79999_iter.pth）：
           前景 = 全部人体语义类别（1..18：脸/眉/眼/鼻/嘴/唇/发/耳/耳饰/颈/颈饰/服饰/头巾/帽…）
           背景 = 纯场景类别 0
       这是真正贴合人脸边界的掩膜，不会再像几何椭圆那样把耳机/头巾/发丝划进前景、
       把下巴/胡须切进背景。带磁盘缓存：同一张图只解析一次。

    2) FaceBackgroundSplit（'geometry'，旧实现）——
       居中羽化椭圆几何先验，作为消融对照保留；同时也是解析失败（画面里检不出人体）时的兜底。
"""
import hashlib
import os
import sys

import cv2
import numpy as np
import torch

# 仓库根目录（d:\\py\\AURA）：本文件位于 baseline/data/ 下，向上三级
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SIMSWAP_ROOT = os.path.join(_REPO_ROOT, "third_party", "SimSwap")
_BISENET_WEIGHTS = os.path.join(_SIMSWAP_ROOT, "parsing_model", "79999_iter.pth")

# BiSeNet 输入预处理（与 SimSwap 推理脚本保持一致）
_PARSE_INPUT = 512
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _load_bisenet(weights: str, device) -> torch.nn.Module:
    """加载 BiSeNet(19 类) 人脸解析网络。

    注意：parsing_model/resnet.py 的 Resnet18.init_weight 会调用 modelzoo.load_url 去下载
    torchvision 预训练权重。这里把 load_url 打桩成返回空 dict（该网络随后会被
    load_state_dict 用人脸解析权重整体覆盖），避免联网、也避免下载失败中断流程。
    """
    import torch.utils.model_zoo as modelzoo

    if _SIMSWAP_ROOT not in sys.path:
        sys.path.append(_SIMSWAP_ROOT)          # 追加而非前置，避免遮蔽标准库/三方包
    modelzoo.load_url = lambda *a, **k: {}        # 阻止联网下载

    from parsing_model.model import BiSeNet
    net = BiSeNet(n_classes=19)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class FaceBackgroundSplit:
    """几何先验掩膜（居中羽化椭圆）。

    数据集（FF++ / CelebAMask）是居中人脸裁剪图，人脸稳定位于画面中央，
    于是直接以画面中心为中心构造一个羽化椭圆作为前景掩膜，其余为背景。

    该“人工先验”稳定、可复现、零依赖；现主要作为消融对照，
    以及 FaceParseMask 解析失败时的兜底。
    """

    def __init__(self, size: int = 128, expand: float = 1.0,
                 center_ratio=(0.50, 0.62), detector=None):
        self.size = size
        # expand：在前景椭圆半轴上的额外放大系数
        self.expand = expand
        # center_ratio：中心先验框的 (宽,高) 占整图的比例（与数据集居中裁剪一致）
        self.center_ratio = center_ratio
        # detector：可选的外部人脸检测器，callable(img_bgr)->(x,y,w,h) 或 None
        self.detector = detector

    # ---------- 人脸框 ----------
    def face_bbox(self, img_bgr: np.ndarray):
        """返回人脸框 (x, y, w, h)：优先用外部检测器，否则用中心几何先验。"""
        if self.detector is not None:
            box = self.detector(img_bgr)
            if box is not None:
                return box
        h, w = img_bgr.shape[:2]
        bw = int(w * self.center_ratio[0])   # 估计人脸的宽
        bh = int(h * self.center_ratio[1])   # 估计人脸的高
        x = int((w - bw) / 2)                # 水平居中
        y = int((h - bh) / 2)                # 垂直居中
        return (x, y, bw, bh)

    # ---------- 掩膜 ----------
    def mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """返回前景掩膜，取值 0~1，形状 (H, W)，边缘做了高斯羽化。"""
        h, w = img_bgr.shape[:2]
        x, y, bw, bh = self.face_bbox(img_bgr)

        mask = np.zeros((h, w), np.float32)
        cx, cy = int(x + bw / 2), int(y + bh / 2)                          # 椭圆中心
        ax, ay = int(bw * self.expand / 2), int(bh * self.expand / 2)      # 半轴
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)          # 实心椭圆

        k = max(3, int(min(h, w) * 0.08) | 1)   # 模糊核（奇数），羽化边缘
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask


class FaceParseMask:
    """BiSeNet 人脸解析掩膜提供器（真实人脸区域 + 磁盘缓存）。

    接口与 FaceBackgroundSplit 一致：mask(img_bgr) -> float32(H, W)。
    额外的 mask_for_path(path, size) 用于数据集/训练，掩膜按 (路径, size) 缓存为
    uint8 的 .npy，同一张图（例如同一真实帧既当真实样本、又当伪造图的源图/供体图）
    在整个实验里只会解析一次。
    """

    def __init__(self, size: int = 128, center_ratio=(0.50, 0.62), expand: float = 1.0,
                 fg_classes=None, cache_dir: str = None, device=None, weights: str = None,
                 fallback_ratio: float = 0.01):
        self.size = int(size)
        # 前景语义类别：默认取 1..18（除 0=背景外的全部人体类别）
        self.fg_classes = tuple(int(c) for c in (fg_classes if fg_classes else range(1, 19)))
        self.cache_dir = cache_dir
        # fallback_ratio：前景占比低于该阈值说明解析没检出人体 -> 退回几何椭圆
        self.fallback_ratio = float(fallback_ratio)
        self.weights = weights or _BISENET_WEIGHTS
        # 解析网络很轻，优先放 GPU；缓存构建完成后用 unload() 释放，避免与 HashVAE 争显存
        want_cuda = torch.cuda.is_available()
        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if want_cuda else "cpu")
        self._fallback = FaceBackgroundSplit(size, expand=expand, center_ratio=center_ratio)
        self._net = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    # ---------- 解析网络 ----------
    def _ensure_net(self) -> torch.nn.Module:
        if self._net is None:
            self._net = _load_bisenet(self.weights, self.device)
        return self._net

    def unload(self) -> None:
        """释放解析网络并清空显存缓存（掩膜缓存构建完成后调用）。"""
        self._net = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _fg_soft(self, img_bgr: np.ndarray) -> np.ndarray:
        """在 512x512 解析尺度上算出 0/1 的前景软掩膜。"""
        net = self._ensure_net()
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (_PARSE_INPUT, _PARSE_INPUT), interpolation=cv2.INTER_LINEAR)
        x = (x.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.device)
        labels = net(x)[0].argmax(1)[0].cpu().numpy()                 # (512,512) 语义标签
        return np.isin(labels, np.asarray(self.fg_classes, np.uint8)).astype(np.float32)

    def _mask_at(self, img_bgr: np.ndarray, size: int) -> np.ndarray:
        """原图 -> (size,size) float32 前景掩膜；解析不出人体时退回几何椭圆。"""
        m = cv2.resize(self._fg_soft(img_bgr), (size, size),
                       interpolation=cv2.INTER_AREA)                  # AREA 下采样自带边缘羽化
        if float(m.mean()) < self.fallback_ratio:
            m = cv2.resize(self._fallback.mask(img_bgr), (size, size),
                           interpolation=cv2.INTER_AREA)
        return np.clip(m, 0.0, 1.0).astype(np.float32)

    # ---------- 对外接口 ----------
    def mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """原图分辨率 (H, W) 的前景掩膜（与 FaceBackgroundSplit 接口一致）。"""
        h, w = img_bgr.shape[:2]
        m = cv2.resize(self._fg_soft(img_bgr), (w, h), interpolation=cv2.INTER_AREA)
        if float(m.mean()) < self.fallback_ratio:
            m = self._fallback.mask(img_bgr)
        return np.clip(m, 0.0, 1.0).astype(np.float32)

    def _cache_key(self, path: str, size: int) -> str:
        raw = f"{os.path.abspath(path)}|{size}|{self.fg_classes}|v1"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def mask_for_path(self, path: str, size: int = None) -> np.ndarray:
        """取 path 对应图片的 (size,size) 前景掩膜，命中磁盘缓存则直接读取。"""
        size = int(size or self.size)
        if not self.cache_dir:
            return self._mask_at(self._read(path), size)
        cache_file = os.path.join(self.cache_dir, self._cache_key(path, size) + ".npy")
        if os.path.exists(cache_file):
            m = np.load(cache_file)
            if m.shape == (size, size):
                return m.astype(np.float32) / 255.0
        m = self._mask_at(self._read(path), size)
        np.save(cache_file, np.round(m * 255.0).astype(np.uint8))
        return m

    def prebuild(self, paths, verbose: bool = True) -> int:
        """批量预热掩膜缓存（去重后逐张解析，已缓存的会直接跳过）。"""
        uniq = sorted({os.path.abspath(p) for p in paths if p})
        for i, p in enumerate(uniq, 1):
            self.mask_for_path(p)
            if verbose and (i % 500 == 0 or i == len(uniq)):
                print(f"  [mask-cache] {i}/{len(uniq)}")
        return len(uniq)

    @staticmethod
    def _read(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        return img
