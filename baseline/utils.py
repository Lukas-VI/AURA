"""AURA 项目通用工具：配置加载、随机种子、设备选择。

论文参考：Deep hashing framework for proactively deepfake retrieval and detection (Liu et al., 2026)
本文件只放被多处复用的轻量函数，避免各脚本重复造轮子。
"""
import os
import random

import numpy as np
import torch
import yaml

# 包根目录（baseline/）：以本文件所在位置为准，保证任何入口都能定位到 configs/ 等资源
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    """把相对路径解析为相对 baseline 包目录的绝对路径；绝对路径原样返回。"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def resolve_path(path: str) -> str:
    """公开版路径解析（不创建目录）。

    读取权重 / history 等已有文件时用这个：无论从哪个工作目录启动脚本，
    都能相对 baseline 包目录正确定位，避免出现“找不到 checkpoints/xxx.pth”。
    """
    return _resolve(path)


def load_config(config_path: str = "configs/baseline.yaml") -> dict:
    """加载配置：先读 default.yaml 作为基础，再用目标配置覆盖同名键。

    这样单个实验文件（baseline.yaml）只需写出与默认值不同的部分。
    """
    with open(_resolve("configs/default.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(_resolve(config_path), "r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    cfg.update(override or {})
    # 数据根目录支持绝对路径（本数据集在 E 盘）
    cfg["data_root"] = cfg["data_root"]
    return cfg


def set_seed(seed: int = 42) -> None:
    """固定所有随机源，保证实验可复现（文档第 12 条要求记录随机种子）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # CPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 多卡也固定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: dict) -> torch.device:
    """选择运行设备：auto 表示有 GPU 用 GPU，否则回退 CPU。"""
    want = str(cfg.get("device", "auto")).lower()
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> str:
    """确保目录存在并返回绝对路径（相对路径相对项目根）。"""
    path = _resolve(path)
    os.makedirs(path, exist_ok=True)
    return path
