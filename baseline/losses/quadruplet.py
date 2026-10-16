"""四元组损失 (Quadruplet Loss)。

论文出处：Liu et al. 2026, 3.4.2 Loss function（式 14、15）。

论文定义（式 15）：
    L_quadruplet = max(0, ‖C_f(q) − C_f(q+1)‖² − ‖C_f(q) − C_f(q−)‖² + α)
                 + max(0, ‖C_b(q) − C_b(q+2)‖² − ‖C_b(q) − C_b(q−)‖² + β)
其中：
    q     —— 查询真实图（源图）
    q+1   —— 在前景上被篡改的版本（本项目中 = 换脸伪造图）
    q+2   —— 在背景上被篡改的版本（本项目中 = 背景攻击图）
    q−    —— 不同类（不同身份）的样本
    α = β = 0.5（论文给定的 margin）
    C_f / C_b 分别是【前景哈希码 / 背景哈希码】

相对旧实现的修正：
    1) 前景项与背景项【分开计算】——前景项只在前景哈希码上算，背景项只在背景哈希码上算，
       而不是把前后景拼成一个哈希后统一算（旧实现正是这样，导致真假不可分）。
    2) 距离用【平方 L2 距离】‖·‖²（论文式 14/15 明确写的是平方），旧实现用了开方的
       pairwise_distance。
"""
import torch
import torch.nn.functional as F


def _sq_l2(a, b):
    """逐样本平方 L2 距离：‖a − b‖²（沿特征维求和，保留 batch 维）。"""
    return ((a - b) ** 2).sum(dim=1)


def quadruplet_loss(h_f_q, h_f_p1, h_b_q, h_b_p2, h_f_neg, h_b_neg,
                    margin_fg: float = 0.5, margin_bg: float = 0.5):
    """前景/背景双项四元组损失。

    参数
    ----
    h_f_q   : 查询(真实源图)的前景哈希特征
    h_f_p1  : 前景被篡改样本(q+1)的前景哈希特征
    h_b_q   : 查询(真实源图)的背景哈希特征
    h_b_p2  : 背景被篡改样本(q+2)的背景哈希特征
    h_f_neg : 负样本(q−)的前景哈希特征（不同身份）
    h_b_neg : 负样本(q−)的背景哈希特征（不同身份）
    margin_fg / margin_bg : α / β，论文取 0.5
    """
    # 前景项（论文式 15 第一行）
    d_fg_pos = _sq_l2(h_f_q, h_f_p1)          # ‖C_f(q) − C_f(q+1)‖²
    d_fg_neg = _sq_l2(h_f_q, h_f_neg)         # ‖C_f(q) − C_f(q−)‖²
    loss_fg = F.relu(d_fg_pos - d_fg_neg + margin_fg)

    # 背景项（论文式 15 第二行）
    d_bg_pos = _sq_l2(h_b_q, h_b_p2)          # ‖C_b(q) − C_b(q+2)‖²
    d_bg_neg = _sq_l2(h_b_q, h_b_neg)         # ‖C_b(q) − C_b(q−)‖²
    loss_bg = F.relu(d_bg_pos - d_bg_neg + margin_bg)

    return (loss_fg + loss_bg).mean()
