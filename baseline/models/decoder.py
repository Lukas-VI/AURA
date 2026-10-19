"""解码器（仅在训练阶段用于重构约束）。

从 hash_dim 维哈希特征出发，先线性扩展到 base_size×base_size 的特征图，再逐级 2 倍上采样
恢复到 image_size×image_size。上采样次数由 image_size 决定：
    128px -> 8x8 起步，4 次翻倍（8→16→32→64→128）
    256px -> 8x8 起步，5 次翻倍（8→16→32→64→128→256）
通道数每翻倍一次减半，最后一级卷积输出 3 通道。
"""
import math

import torch
import torch.nn as nn


class HashDecoder(nn.Module):
    """哈希特征 -> 重构图像 (B,3,H,W)。"""

    def __init__(self, hash_dim: int = 128, out_channels: int = 3, image_size: int = 128,
                 base_size: int = 8, base_channels: int = 512):
        super().__init__()
        self.base_size = base_size
        self.base_channels = base_channels

        # 需要翻倍的次数：base_size * 2^U = image_size
        self.num_ups = int(round(math.log2(image_size / base_size)))
        if base_size * (2 ** self.num_ups) != image_size:
            raise ValueError(f"image_size={image_size} 必须是 base_size={base_size} 的 2 次幂倍数")

        # 线性扩展：hash_dim -> base_channels * base_size * base_size
        self.fc = nn.Linear(hash_dim, base_channels * base_size * base_size)

        # 前 U-1 次为“卷积 + 激活 + 上采样”，第 U 次为“卷积到 3 通道 + 上采样 + Sigmoid”
        blocks, in_ch = [], base_channels
        for _ in range(self.num_ups - 1):
            out_ch = max(16, in_ch // 2)
            blocks.append(self._block(in_ch, out_ch))
            in_ch = out_ch
        blocks.append(nn.Conv2d(in_ch, out_channels, 3, padding=1))
        blocks.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        blocks.append(nn.Sigmoid())                                    # 输出 0~1
        self.upsample = nn.Sequential(*blocks)

    @staticmethod
    def _block(in_ch, out_ch):
        """一层“卷积 + 激活 + 2 倍上采样”的组合。"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    def forward(self, hash_feat):
        x = self.fc(hash_feat)                                       # (B, C*8*8)
        x = x.view(-1, self.base_channels, self.base_size, self.base_size)  # 还原成特征图
        return self.upsample(x)                                      # (B, 3, image_size, image_size)

