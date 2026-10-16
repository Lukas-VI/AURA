"""解码器（仅在训练阶段用于重构约束）。

从 hash_dim 维哈希特征出发，先线性扩展到 8x8 的特征图，再逐级上采样恢复到 128x128。
注：文档示例以 4x4 为起点、上采样 4 次只能到 64x64，与 128 的输入尺寸不匹配，
这里改为 8x8 起点、上采样 4 次恰好得到 128x128。
"""
import torch
import torch.nn as nn


class HashDecoder(nn.Module):
    """哈希特征 -> 重构图像 (B,3,H,W)。"""

    def __init__(self, hash_dim: int = 128, out_channels: int = 3, base_size: int = 8,
                 base_channels: int = 512):
        super().__init__()
        self.base_size = base_size
        self.base_channels = base_channels

        # 线性扩展：hash_dim -> base_channels * base_size * base_size
        self.fc = nn.Linear(hash_dim, base_channels * base_size * base_size)

        # 4 次上采样：8 -> 16 -> 32 -> 64 -> 128
        self.upsample = nn.Sequential(
            self._block(base_channels, base_channels // 2),  # 8x8  -> 16x16
            self._block(base_channels // 2, base_channels // 4),  # 16   -> 32
            self._block(base_channels // 4, base_channels // 8),  # 32   -> 64
            nn.Conv2d(base_channels // 8, out_channels, 3, padding=1),  # 64 -> 128（不升采样）
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Sigmoid(),                                                  # 输出 0~1
        )

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
        return self.upsample(x)                                      # (B, 3, 128, 128)
