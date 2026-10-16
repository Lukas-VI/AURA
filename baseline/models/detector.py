"""Deepfake 检测块 (DDB)。

论文出处：Liu et al., "Deep hashing framework for proactively deepfake retrieval and
detection", 3.4.1 Deepfake detection block（式 12、13）。

关键点（相对旧实现的修正）：
    1) DDB 的输入是【前景哈希特征 F_f,hash】，而【不是】前后景拼接后的完整哈希。
       论文原文：“the Deepfake detection block takes the foreground hash feature F_f,hash
       as input, and determines the image authenticity as output.”（3.4.1）
    2) 论文式 12 里，每个全连接层的输入是“前面所有层输出的聚合”：
           ĥ_k = H_cls^k( Σ_{j=0}^{k-1} out_j )，out_0 = F_f,hash
       因此这里用“累加聚合”的稠密式多层感知机，而不是普通串联 MLP。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepfakeDetector(nn.Module):
    """前景哈希特征 (B, K) -> 伪造概率 (B, 1)。

    结构对应论文式 12：
        out_0 = h_f
        out_k = LeakyReLU(Linear_k( Σ_{j<k} out_j ))     # 每层吃“前面所有输出的和”
        pred  = Sigmoid(Linear_cls( Σ_k out_k ))          # 分类聚合
    """

    def __init__(self, hash_dim: int = 64, hidden_dim: int = None, num_layers: int = 3):
        super().__init__()
        hidden_dim = int(hidden_dim or hash_dim)   # 各线性层输出维度需一致，才能“聚合求和”
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        for _ in range(num_layers):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        # 分类层：输入是“所有层输出之和”（各层输出维相同，可直接相加）
        self.cls = nn.Linear(hidden_dim, 1)

    def forward(self, h_f):
        outs = [h_f]                              # out_0 = 前景哈希特征 (B,K)
        for lin in self.linears:
            agg = torch.stack(outs, dim=0).sum(dim=0)   # Σ 之前所有输出（论文式 12）
            outs.append(F.leaky_relu(lin(agg), 0.1))
        out_k = torch.stack(outs[1:], dim=0).sum(dim=0)  # Σ_k out_k
        return torch.sigmoid(self.cls(out_k))            # (B, 1) 伪造概率
