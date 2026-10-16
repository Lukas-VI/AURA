"""完整模型 HashVAE（简化版，按论文结构修正）。

论文出处：Liu et al. 2026, 3.2 / 3.3.2 / 3.4.1。

论文的核心设计（本文件据此修正）：
    1) 预处理把输入图拆成前景 x_f（人脸区域）与背景 x_b，两者【分别】送入编码器，
       且【共享编码器参数】（论文 3.3.2：“the parameters of foreground encoder and
       background encoder are shared”）。
    2) 编码器分别输出前景哈希特征 F_f,hash 与背景哈希特征 F_b,hash；
       哈希码 C = C_f + C_b（前后景哈希码拼接，论文式 8 之后一段）。
    3) 解码器同样共享：~x_f = H_D(F_f,hash)、~x_b = H_D(F_b,hash)，分别重构前景/背景。
    4) 检测块 DDB 只吃【前景哈希特征 F_f,hash】（论文 3.4.1）。

相对旧实现的修正：旧实现把整图编码成单一哈希，前后景混在一起，检测块也读这个混合哈希，
导致真假样本哈希不可分（cls 恒为 ln2）。现按论文改为前后景分离编码 + DDB 只读前景哈希。

简化说明（详见 README 差异表）：
    - 前景/背景分离用【逐样本的前景掩膜】（由数据集用 BiSeNet 人脸解析得到，见
      data/preprocessing.py FaceParseMask），而非论文的人脸检测 + 注意力擦除。
      模型内置的固定几何掩膜 self.fg_mask 仅作为未传掩膜时的兜底 / 消融对照。
    - 论文式 7 用 softmax 得到哈希特征；若按字面在哈希维上做 softmax，所有比特会挤成
      一个 one-hot（Σ=1），哈希码坍缩。故这里沿用“逐比特映射到 (0,1)”的稳定实现
      （BN + Tanh 再线性映射），保证软量化/最小熵可用。
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

from .decoder import HashDecoder
from .detector import DeepfakeDetector
from .encoder import HashCode, HashEncoder


def build_fg_mask(image_size: int, center_ratio=(0.50, 0.62), expand: float = 1.0,
                  blur_ratio: float = 0.08) -> torch.Tensor:
    """构造前景掩膜（居中羽化椭圆），返回 (1,1,H,W) 的 float 张量。

    与数据集里的 FaceBackgroundSplit 使用同一套几何先验（center_ratio/expand），
    保证模型内部的“前后景分离”与伪造图合成时用的区域一致。
    """
    mask = np.zeros((image_size, image_size), np.float32)
    bw = image_size * center_ratio[0]      # 前景框宽
    bh = image_size * center_ratio[1]      # 前景框高
    cx = cy = image_size // 2
    ax, ay = int(bw * expand / 2), int(bh * expand / 2)   # 椭圆半轴
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
    k = max(3, int(image_size * blur_ratio) | 1)          # 奇数模糊核，边缘羽化
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return torch.from_numpy(mask)[None, None]             # (1,1,H,W)


class HashVAE(nn.Module):
    """编码器(共享) + 解码器(共享) + 检测块 + 哈希量化。"""

    def __init__(self, hash_dim: int = 128, base_channels: int = 64,
                 num_branches: int = 2, num_rb: int = 2, num_scales: int = 2,
                 aff_channels: int = 512, image_size: int = 128,
                 center_ratio=(0.50, 0.62), expand: float = 1.0):
        super().__init__()
        # 论文里 C = C_f + C_b，即总码长 = 前景码长 + 背景码长；这里对半分
        self.hash_dim = int(hash_dim)              # 总哈希码长度
        branch_dim = max(1, self.hash_dim // 2)    # 前景/背景各自的码长 K

        # 共享编码器：分别作用于前景、背景
        self.encoder = HashEncoder(in_channels=3, base_channels=base_channels,
                                   num_branches=num_branches, num_rb=num_rb,
                                   num_scales=num_scales, hash_dim=branch_dim,
                                   aff_channels=aff_channels)
        # 共享解码器：分别重构前景、背景
        self.decoder = HashDecoder(hash_dim=branch_dim)
        # 检测块：只吃前景哈希特征
        self.detector = DeepfakeDetector(hash_dim=branch_dim)
        self.hash_code = HashCode()                # 哈希量化（训练软量化/推理硬量化）

        # 固定的前后景掩膜（buffer 会随模型一起搬到 GPU）
        self.register_buffer("fg_mask", build_fg_mask(image_size, center_ratio, expand))

    # ---------- 只编码（不重构/不检测），供训练阶段 2 的负样本等节省显存 ----------
    def encode(self, x, mask=None):
        """输入 (B,3,H,W) 与可选前景掩膜 (B,1,H,W)，返回前后景哈希特征/哈希码等。

        mask 为逐样本的前景掩膜（由数据集按真实人脸解析得到）；为 None 时退回到模型内置的
        固定几何掩膜 self.fg_mask（旧基线行为 / 兜底），保证 4GB 显存下仍可跑通。
        """
        if mask is None:
            fg_mask = self.fg_mask                             # (1,1,H,W) 广播
        else:
            fg_mask = mask if mask.dim() == 4 else mask.unsqueeze(1)   # (B,1,H,W)
            fg_mask = fg_mask.to(dtype=x.dtype, device=x.device)
        fg = x * fg_mask                           # 前景（人脸区域）
        bg = x * (1.0 - fg_mask)                   # 背景
        h_f = self.encoder(fg)                     # 前景哈希特征 F_f,hash (B,K)
        h_b = self.encoder(bg)                     # 背景哈希特征 F_b,hash (B,K)
        return {
            "fg": fg, "bg": bg, "mask": fg_mask,
            "h_f": h_f, "h_b": h_b,
            "hash": torch.cat([h_f, h_b], dim=1),          # 连续哈希特征（供最小熵/平衡）
            "code_f": self.hash_code(h_f),                 # 前景哈希码 C_f
            "code_b": self.hash_code(h_b),                 # 背景哈希码 C_b
        }

    def forward(self, x, mask=None, need_recon: bool = True, need_pred: bool = True):
        """完整前向。返回 dict，按需附带重构/预测以节省 4GB 显存下的开销。"""
        out = self.encode(x, mask=mask)
        out["code"] = torch.cat([out["code_f"], out["code_b"]], dim=1)   # C = C_f + C_b
        if need_recon:
            out["recon_f"] = self.decoder(out["h_f"])   # ~x_f
            out["recon_b"] = self.decoder(out["h_b"])   # ~x_b
        if need_pred:
            out["pred"] = self.detector(out["h_f"])     # 只读前景哈希特征
        return out

    @staticmethod
    def build_from_config(cfg: dict) -> "HashVAE":
        """按配置字典构建模型，便于训练/测试脚本复用。"""
        return HashVAE(
            hash_dim=int(cfg["hash_dim"]),
            base_channels=int(cfg.get("base_channels", 64)),
            num_branches=int(cfg.get("num_branches", 2)),
            num_rb=int(cfg.get("num_rb", 2)),
            num_scales=int(cfg.get("num_scales", 2)),
            aff_channels=int(cfg.get("aff_channels", 512)),
            image_size=int(cfg.get("image_size", 128)),
            center_ratio=tuple(cfg.get("fg_center_ratio", [0.50, 0.62])),
            expand=float(cfg.get("fg_expand", 1.0)),
        )
