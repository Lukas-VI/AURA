"""哈希编码器（简化版 HashVAE 的编码分支）。

结构：初始卷积 -> 多尺度多分支残差提取 -> 注意力特征融合(AFF) -> 哈希特征投影。
输出为 hash_dim 维、取值 0~1 的连续哈希特征（训练时用于软量化）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """基础残差块：2 个 3x3 卷积 + LeakyReLU，带短残差连接。"""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)   # 第一层卷积
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)   # 第二层卷积
        self.act = nn.LeakyReLU(0.1)                               # 激活函数

    def forward(self, x):
        residual = x                        # 记录输入，供残差相加
        out = self.act(self.conv1(x))       # 卷积 + 激活
        out = self.conv2(out)               # 卷积（先不加激活）
        return self.act(out + residual)     # 残差相加后再激活


class MultiBranchBlock(nn.Module):
    """多分支残差块（简化版：默认 2 分支 x 2 RB）。

    分支之间做“前 m-1 个分支输出融合”，最后用可学习权重加权求和，并加长残差。
    """

    def __init__(self, channels: int = 64, num_branches: int = 2, num_rb: int = 2):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList()
        for _ in range(num_branches):
            # 每个分支是 num_rb 个串联的残差块
            branch = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_rb)])
            self.branches.append(branch)
        self.fusion_weights = nn.Parameter(torch.ones(num_branches))  # 可学习的跨分支融合权重

    def forward(self, x):
        outputs = []
        for m, branch in enumerate(self.branches):
            if m == 0:
                out = branch(x)                                   # 第一个分支直接作用于输入
            else:
                prev = torch.stack(outputs, dim=0).mean(dim=0)   # 前面分支输出的平均
                out = branch(x + prev)                            # 融合前分支信息再送进当前分支
            outputs.append(out)

        weights = F.softmax(self.fusion_weights, dim=0)           # 权重归一化
        out = sum(w * o for w, o in zip(weights, outputs))        # 加权融合
        return out + x                                            # 长残差连接


class AttentionFeatureFusion(nn.Module):
    """注意力特征融合(AFF)：把不同尺度特征对齐后按注意力权重加权求和。"""

    def __init__(self, channels_list, out_channels: int = 512):
        super().__init__()
        # 每个输入尺度用 1x1 卷积调整到统一通道数
        self.adjust_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in channels_list
        ])
        # 注意力分支：拼接特征 -> 每尺度一个权重 -> Sigmoid 到 (0,1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * len(channels_list), len(channels_list), 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        adjusted = [conv(f) for conv, f in zip(self.adjust_convs, features)]  # 统一通道
        target_size = adjusted[0].shape[-2:]                                   # 以第一个尺度为基准尺寸
        adjusted = [F.interpolate(a, size=target_size, mode="bilinear", align_corners=False)
                    for a in adjusted]                                          # 对齐空间尺寸
        concat = torch.cat(adjusted, dim=1)                                     # 通道维拼接
        weights = self.attention(concat)                                        # (B, S, H, W)
        # 每个尺度乘上自己的注意力权重后求和；w 已是 (B,1,H,W)，可直接与 (B,C,H,W) 广播
        out = sum(w * a for w, a in zip(weights.split(1, dim=1), adjusted))
        return out


class HashEncoder(nn.Module):
    """哈希编码器：图像 (B,3,H,W) -> 哈希特征 (B,hash_dim)。"""

    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_branches: int = 2, num_rb: int = 2, num_scales: int = 2,
                 hash_dim: int = 128, aff_channels: int = 512):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, base_channels, 3, padding=1)  # 初始卷积

        self.scales = nn.ModuleList()      # 各尺度上的多分支块
        self.downsample = nn.ModuleList()  # 尺度间的降采样层
        channels_list = []
        for s in range(num_scales):
            channels = base_channels * (2 ** s)   # 尺度越深通道越多
            channels_list.append(channels)
            self.scales.append(MultiBranchBlock(channels, num_branches, num_rb))
            if s < num_scales - 1:                # 最后一个尺度不降采样
                self.downsample.append(
                    nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
                )

        self.aff = AttentionFeatureFusion(channels_list, out_channels=aff_channels)  # 注意力融合
        # 哈希特征投影：卷积 -> 激活 -> 全局池化 -> 全连接 -> BN -> Tanh(-1~1)
        # 说明：原设计用 Sigmoid 直接输出 (0,1)，实测在“最小熵损失”下第 2 轮即饱和到
        #       0/1、梯度消失，导致所有样本被编码成同一个码（哈希坍缩）。
        #       这里改为 Tanh + BatchNorm1d：中间区梯度更大、分布更稳，再把 (-1,1) 映射到 (0,1)。
        self.hash_proj = nn.Sequential(
            nn.Conv2d(aff_channels, aff_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(aff_channels, hash_dim),
            nn.BatchNorm1d(hash_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv0(x)                  # 初始特征
        features = []
        for s, mb in enumerate(self.scales):
            x = mb(x)                      # 多分支残差提取
            features.append(x)             # 收集当前尺度特征
            if s < len(self.downsample):
                x = self.downsample[s](x)  # 进入下一尺度前降采样
        fused = self.aff(features)         # 注意力融合
        logits = self.hash_proj(fused)     # (B, hash_dim)，Tanh 后取值在 (-1, 1)
        return (logits + 1.0) / 2.0        # 映射到 (0, 1)，供软量化 / 最小熵 / 硬量化使用


class HashCode(nn.Module):
    """哈希码量化：训练时软量化（直接用连续值），推理时硬量化（阈值 0.5）。"""

    def forward(self, hash_feat):
        if self.training:
            return hash_feat                                 # 软量化，保证梯度可回传
        return (hash_feat > 0.5).float()                     # 硬量化成 0/1 比特串
