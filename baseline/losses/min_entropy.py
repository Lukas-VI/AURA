"""最小熵比特损失：驱使哈希特征逼近 0 或 1，得到“类二值”的紧凑编码。

公式：L_h = -1/N * Σ [ h*log(h) + (1-h)*log(1-h) ]
当 h 接近 0 或 1 时熵最小，损失最小；h=0.5 时熵最大。

另附“比特平衡”损失：只靠最小熵会把所有样本压成同一个码（哈希坍缩），
因此额外要求批次内每个比特的均值接近 0.5，保证不同样本拿到不同比特。
"""
import torch


def min_entropy_loss(hash_feat):
    eps = 1e-7                                            # 防止 log(0)
    h = hash_feat.clamp(eps, 1 - eps)                     # 把取值夹到 (0,1)
    entropy = -(h * torch.log(h) + (1 - h) * torch.log(1 - h))  # 逐元素二元熵
    return entropy.mean()                                 # 对全 batch、全比特取平均


def bit_balance_loss(hash_feat):
    """比特平衡损失：要求同一批次内“每个比特的均值”接近 0.5。

    若编码器把所有样本压成同一个码（哈希坍缩），该均值会远离 0.5，
    此项随即变大并给出梯度，把编码拉回“不同样本产生不同比特”的状态。
    """
    return ((hash_feat.mean(dim=0) - 0.5) ** 2).mean()

