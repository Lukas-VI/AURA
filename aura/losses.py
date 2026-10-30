"""对比学习损失：监督对比（SupCon，带跨批内存）+ 防坍缩方差项。"""
import torch


def supcon_loss(z, labels, queue_z=None, queue_labels=None, tau: float = 0.1):
    """监督对比损失（Khosla et al. 2020）的"跨批内存"版本（Wang et al. 2020, XBM）。

    z             : (N, D) 已 L2 归一化的当前批嵌入（带梯度）
    labels        : (N,) 同源身份标签（同源 = 同一段真实视频 X）
    queue_z/queue_labels : 历史批的嵌入与标签（已 detach）；None 表示只用批内
    tau           : 温度

    正样本 = 标签相同的其它嵌入（批内 + 内存里历史出现过的同源样本）；
    负样本 = 标签不同的所有嵌入。内存队列让 batch 只有个位数时也能有数千个负样本 ——
    在 22GB 卡上 256px 全规格只能开 batch≈8，靠扩展队列才凑得出可用的对比信号。
    """
    keys = z if queue_z is None else torch.cat([z, queue_z], dim=0)
    key_labels = labels if queue_labels is None else torch.cat([labels, queue_labels], dim=0)
    n = z.shape[0]

    logits = (z @ keys.t()) / tau                        # (N, N+Q)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()   # 数值稳定

    # 把自己那一条 key 从分子分母里剔除：否则每行都有一个自匹配的"完美正样本"，
    # 该行损失被平凡解拉低，学不到东西。
    self_mask = torch.zeros_like(logits, dtype=torch.bool)
    self_mask[:, :n] = torch.eye(n, dtype=torch.bool, device=z.device)
    pos_mask = (labels[:, None] == key_labels[None, :]) & (~self_mask)
    logits = logits.masked_fill(self_mask, -1e4)

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if not bool(valid.any()):
        return z.sum() * 0.0                             # 整批无正样本：返回 0 但保留计算图
    per_anchor = -(pos_mask * log_prob).sum(dim=1) / n_pos.clamp(min=1)
    return per_anchor[valid].mean()


def grid_supcon_loss(cells, labels, tau: float = 0.1):
    """网格级监督对比损失：在每个空间位置上各自做一次 SupCon。

    动机（第 5.3 节证据 + 8 选 1 诊断）：整图对比学习只要求"整张图 -> 一个向量"可分，
    同一视频相邻帧的差异在池化/线性混合后被平均掉；而 \code{r8\_norm} 之所以能到 60%，
    正是因为它保留了 8×8 的逐格灰度。这里把 head 之前的 8×8 网格按【位置】拆开，
    在同一位置上做跨样本对比：同一源帧（不同伪造方式）应当靠近，同视频其他帧应当远离。
    于是模型被直接要求"每个格子都具备区分近重复帧的能力"，而不是把这件事交给一个全局向量。

    cells  : (N, G*G, D) 已逐位置 L2 归一化的网格描述子（N=批内样本数）
    labels : (N,) 与主对比损失相同的帧级标签（source_id）
    """
    n, g2, d = cells.shape
    sim = torch.einsum("ngd,mgd->gmn", cells, cells) / tau      # (G², N, N)
    self_mask = torch.eye(n, dtype=torch.bool, device=cells.device)
    pos_mask = ((labels[:, None] == labels[None, :]) & (~self_mask)).expand(g2, n, n)
    logits = sim.masked_fill(self_mask[None], -1e4)

    log_prob = logits - torch.logsumexp(logits, dim=2, keepdim=True)
    n_pos = pos_mask.sum(dim=2)                                  # (G², N)
    valid = n_pos > 0
    if not bool(valid.any()):
        return cells.sum() * 0.0                                 # 无正样本：返回 0 但保留计算图
    per_anchor = -(pos_mask * log_prob).sum(dim=2) / n_pos.clamp(min=1)
    return per_anchor[valid].mean()


def variance_loss(z, gamma: float = None, eps: float = 1e-4):
    """VICReg 式方差铰链项：要求每个维度在批内的标准差至少为 gamma，防止嵌入坍缩。

    z 传入【已 L2 归一化】的嵌入：均匀分布在单位球面上时各维 std ≈ 1/sqrt(D)，
    故 gamma 默认取 1/sqrt(D)。该项是对比损失的兜底保险（对比损失本身通常已能防坍缩）。
    """
    if gamma is None:
        gamma = 1.0 / (z.shape[1] ** 0.5)
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.relu(gamma - std).mean()
