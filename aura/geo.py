"""可微几何验证模块（AURA-Geo / K 轮）。

用途：**嵌入 AURA 端到端训练**，而不是外挂检索期重排器。用真实检索对
（退化伪造查询 ↔ 同源干净真帧）做几何一致性监督，梯度经共享主干（conv0 + scales[0]）回传，
让"背景可被同一单应解释"这件事直接塑形检索表征。

全链可微（无 SIFT / 无 RANSAC 的任何不可微算子）：
    软互最近邻匹配（softmax 替代硬 argmax + Lowe ratio）
      -> 软期望对应点 b_hat = Σ_j w_ij · coord_j
      -> 加权可微 DLT（torch.linalg.lstsq，固定 h33=1）
      -> 软内点占比 s=exp(-r²/2σ²) 替代 RANSAC 硬内点判定

两处历史 bug 已修（见 aura/runs/_probe_diff_geom.py 注释）：
    ① DLT 系数行符号须与 RHS 一致（否则等价于拟合 a→-b，自配对也会给近零分）；
    ② 系数行 stack 的轴须为 dim=2（[r1_N, r2_N] 行序），用 dim=3 会按列交错、矩阵退化 rank=7。
"""
import torch
import torch.nn.functional as F


def grid_coords(grid: int, device) -> torch.Tensor:
    """G×G 网格点的归一化坐标 [-1,1]（(x,y)，列优先 k = j*G + i，与特征图 flatten 顺序一致）。"""
    us = (torch.arange(grid, device=device).float() + 0.5) / grid * 2 - 1
    vs = (torch.arange(grid, device=device).float() + 0.5) / grid * 2 - 1
    x = us[None, :].repeat(grid, 1).reshape(-1)             # i 快速变化
    y = vs[:, None].repeat(1, grid).reshape(-1)
    return torch.stack([x, y], dim=1)                       # (N,2)


def soft_mutual_match(Fa: torch.Tensor, Fb: torch.Tensor, temp: float) -> torch.Tensor:
    """软互最近邻权重：wa=softmax_b(相关体), wb=softmax_a(相关体), w=wa*wb。返回 (B,N,N)。"""
    corr = torch.bmm(Fa.transpose(1, 2), Fb)                # (B,N,N)
    wa = torch.softmax(corr / temp, dim=2)                  # 每个 a 在 b 上的分布
    wb = torch.softmax(corr / temp, dim=1)                  # 每个 b 在 a 上的分布
    return wa * wb


def diff_dlt(a_pts: torch.Tensor, b_pts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """加权可微 DLT（固定 h33=1）：A(2N,8)->最小二乘解 h(8)。a_pts,b_pts (B,N,2)，w (B,N,N)。"""
    w = weights.max(dim=2).values.clamp_min(1e-6)           # (B,N) 每个 a 点的匹配置信度
    x, y = a_pts[..., 0], a_pts[..., 1]
    xp, yp = b_pts[..., 0], b_pts[..., 1]
    z = torch.zeros_like(x)
    one = torch.ones_like(x)
    # 行1: [x, y, 1, 0,0,0, -x*xp, -y*xp] = xp ; 行2: [0,0,0, x, y, 1, -x*yp, -y*yp] = yp
    r1 = torch.stack([x, y, one, z, z, z, -x * xp, -y * xp], dim=2)
    r2 = torch.stack([z, z, z, x, y, one, -x * yp, -y * yp], dim=2)
    A = torch.stack([r1, r2], dim=2).reshape(a_pts.shape[0], -1, 8)   # (B,2N,8)
    t = torch.stack([xp, yp], dim=2).reshape(a_pts.shape[0], -1)      # (B,2N)
    sw = torch.sqrt(w).repeat_interleave(2, dim=1).unsqueeze(2)       # 行权重 sqrt(w)
    As = A * sw
    ts = (t * sw.squeeze(2)).unsqueeze(2)
    # 正规方程解（8×8 批解）：与 lstsq 的加权最小二乘解等价，但快 100 倍以上。
    # lstsq 在 (1600,512,8) 这种批形上实测 543ms，是粗网格几何初筛的绝对瓶颈。
    Ms = As.transpose(1, 2) @ As
    Ms = Ms + 1e-6 * torch.eye(8, device=Ms.device, dtype=Ms.dtype)   # 极小岭，防奇异
    sol = torch.linalg.solve(Ms, As.transpose(1, 2) @ ts)
    return sol.squeeze(2)                                             # (B,8)


def geom_score(h: torch.Tensor, a_pts: torch.Tensor, b_pts: torch.Tensor,
               weights: torch.Tensor, sigma: float) -> torch.Tensor:
    """软内点分数：a 点经 h 变换后与 b 点的残差 r，软内点 s=exp(-r²/2σ²)，加权占比作分 (B,)。"""
    x, y = a_pts[..., 0], a_pts[..., 1]
    xp, yp = b_pts[..., 0], b_pts[..., 1]
    denom = (h[:, 6:7] * x + h[:, 7:8] * y + 1.0).clamp_min(1e-6)
    xt = (h[:, 0:1] * x + h[:, 1:2] * y + h[:, 2:3]) / denom
    yt = (h[:, 3:4] * x + h[:, 4:5] * y + h[:, 5:6]) / denom
    r2 = (xt - xp) ** 2 + (yt - yp) ** 2
    s = torch.exp(-r2 / (2 * sigma ** 2))                   # 软内点
    w = weights.max(dim=2).values.clamp_min(1e-6)           # (B,N)
    return (w * s).sum(1) / w.sum(1)                        # (B,) ∈ (0,1)


def pair_score(Fa: torch.Tensor, Fb: torch.Tensor, coords: torch.Tensor,
               temp: float = 0.05, sigma: float = 0.08) -> torch.Tensor:
    """一对逐位置特征图 (B,C,N) -> 几何一致性分数 (B,)。

    注意：query 与候选在几何退化下并未对齐，对应关系全靠软匹配 w 建立 —— 不能用同一套坐标
    冒充 b 点（那样等于假设恒等变换）。这里用软期望坐标 b_hat_i = Σ_j w_ij·coord_j 作为 b 点。
    """
    b = Fa.shape[0]
    c = coords.unsqueeze(0).expand(b, -1, -1)               # (B,N,2) query 网格坐标
    w = soft_mutual_match(Fa, Fb, temp)                     # (B,N,N)
    wnorm = w / w.sum(dim=2, keepdim=True).clamp_min(1e-6)
    b_hat = torch.bmm(wnorm, c)                             # (B,N,2) 每个 a 的软期望匹配点
    h = diff_dlt(c, b_hat, w)
    return geom_score(h, c, b_hat, w, sigma)


def pair_score_dense(Fa: torch.Tensor, Fb: torch.Tensor, coords: torch.Tensor,
                     temp: float = 0.05, sigma: float = 0.08):
    """同 pair_score，但同时返回【逐位置】的软内点 s 与匹配权重 w（各 (B,N)）。

    为什么要暴露逐位置量：帧级检索里正样本与它的兄弟帧（同视频 k±1）背景逐像素相同，
    绝对几何分（内点占比）对二者都接近满分，判别力全被背景吃掉了。要拿回判别力，必须在
    **候选集内部**逐位置比较 —— 见 aura/runs/_eval_within_video.py 的 margin/var 变体。
    残差公式与 geom_score 完全一致（同一 h、同一 s、同一 w），因此 s 可直接复用。
    """
    b = Fa.shape[0]
    c = coords.unsqueeze(0).expand(b, -1, -1)
    w = soft_mutual_match(Fa, Fb, temp)
    wnorm = w / w.sum(dim=2, keepdim=True).clamp_min(1e-6)
    b_hat = torch.bmm(wnorm, c)
    h = diff_dlt(c, b_hat, w)
    x, y = c[..., 0], c[..., 1]
    xp, yp = b_hat[..., 0], b_hat[..., 1]
    denom = (h[:, 6:7] * x + h[:, 7:8] * y + 1.0).clamp_min(1e-6)
    xt = (h[:, 0:1] * x + h[:, 1:2] * y + h[:, 2:3]) / denom
    yt = (h[:, 3:4] * x + h[:, 4:5] * y + h[:, 5:6]) / denom
    r2 = (xt - xp) ** 2 + (yt - yp) ** 2
    s = torch.exp(-r2 / (2 * sigma ** 2))
    wp = w.max(dim=2).values.clamp_min(1e-6)
    return (wp * s).sum(1) / wp.sum(1), s, wp


def geo_consistency_loss(s_pos: torch.Tensor, s_neg: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """铰链：同源正对分数要高、异源负对分数要低（0 <= s <= 1）。"""
    return F.relu(margin + s_neg - s_pos).mean()
