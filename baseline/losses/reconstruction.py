"""重构损失：约束哈希特征保留足够信息以重构原图（类似 VAE 的重建项）。"""
import torch.nn.functional as F


def reconstruction_loss(x, recon):
    """L2（MSE）重构损失：x 为原图，recon 为解码器输出，二者形状一致。"""
    return F.mse_loss(recon, x)
