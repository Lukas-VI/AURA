"""数据增强。

数据集本身是 256x256 的居中人脸图，图像尺寸统一在 dataset 里用 cv2 缩放到 image_size
（128）；这里只负责“张量化”，避免多次插值造成失真。

注意：训练期的随机水平翻转【不在这里做】——翻转必须与前景掩膜同步进行，
因此在 AURADataset / QuadrupletDataset 读图时成对完成（见 dataset.py）。
"""
from torchvision import transforms


def build_transform():
    """唯一的图像变换：numpy(H,W,3) -> (0,1) 的 CHW 张量。"""
    return transforms.Compose([
        transforms.ToTensor(),
    ])
