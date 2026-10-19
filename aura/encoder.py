"""AURA 自建单编码器：整图输入 -> 连续嵌入（全图对比学习用）。

这是"另起炉灶"的自建模型，与 baseline 复现的 HashVAE 的区别：

1) 只有一个编码器，整图直接输入；不做前景/背景分离、不做 VAE 重构、不做检测头。
2) 输出连续嵌入（不接 Tanh、不做比特量化），训练用监督对比损失；
   检索时既可 L2 归一化后按余弦排序，也可按符号二值化成哈希码按汉明排序。

主干沿用 baseline 里已验证可跑通的多尺度多分支 + AFF 结构，以保留论文"全量规格"
（256px / M=5 / D=5 / 64ch / aff512 / embed2048）的可比性；AFF 段强制 fp32 的
防溢出修复一并保留（根因见 baseline/models/encoder.py 的 hash_proj 注释）。
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from baseline.models.encoder import AttentionFeatureFusion, MultiBranchBlock


class SourceEncoder(nn.Module):
    """整图 (B,3,H,W) -> 嵌入 (B, embed_dim)（未归一化，归一化由损失/评估侧做）。"""

    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_branches: int = 5, num_rb: int = 5, num_scales: int = 2,
                 aff_channels: int = 512, embed_dim: int = 2048,
                 grad_checkpoint: bool = True, use_multiscale_skip: bool = False,
                 spatial_head: bool = False, pool_grid: int = 8):
        super().__init__()
        # 梯度检查点：前向不保存中间激活、反向重算，用来把 256px 全规格的激活显存砍半。
        # 必须开 —— M=5 分支 × D=5 残差块每层都留中间张量供 backward，是显存爆掉的主因。
        self.grad_checkpoint = bool(grad_checkpoint)
        self.conv0 = nn.Conv2d(in_channels, base_channels, 3, padding=1)   # 初始卷积

        self.scales = nn.ModuleList()      # 各尺度上的多分支残差块
        self.downsample = nn.ModuleList()  # 尺度间的降采样层
        channels_list = []
        for s in range(num_scales):
            channels = base_channels * (2 ** s)
            channels_list.append(channels)
            self.scales.append(MultiBranchBlock(channels, num_branches, num_rb))
            if s < num_scales - 1:
                self.downsample.append(nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1))

        # 多尺度低层跳连（"半个 U-Net" 的编码器侧）：每个尺度在【降采样之前】的特征图上做全局
        # 池化再投影成一条 lateral，与 AFF 输出拼接后一起进 head。
        # 动机：零训练实测深度嵌入单独只有帧级 47.75%，而纯 32×32 缩略图有 58.71% —— 主干在逐级
        # 下采样 + 全局平均池化中把低层信息（颜色/纹理/场景布局）丢得差不多了，而检索答案恰恰靠
        # 这些。这里用【可学的 lateral】而不是把原始像素拼进来：伪造帧与其正确源真实帧的背景逐
        # 像素相同（FF++ 的 X_Y 里 X 就是背景来源视频），高分辨率原始像素等同于背景模板匹配，
        # 会把指标变成评测协议自带的像素泄漏，而不是模型真学会了"来源"。
        self.lateral = nn.ModuleList()
        if use_multiscale_skip:
            for c in channels_list:
                self.lateral.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(c, aff_channels), nn.LeakyReLU(0.1)))
        self.aff = AttentionFeatureFusion(channels_list, out_channels=aff_channels)
        # 池化段（两种模式，由 spatial_head 选择）：
        #   GAP 模式（默认，E/G 轮）：卷积 -> 激活 -> 全局平均池化(1×1) -> 展平，输出 aff_channels。
        #       代价：把 AFF 的 256×256 空间图压成 1 个均值（65536:1 的多对一满射），空间排列归零
        #       —— 而 r8_norm（8×8 逐图标准化缩略图，帧级 60.05%）恰恰靠这条粗空间布局取胜，
        #       见 docs/AURA_HANDOFF.md §3.8/§3.9：瓶颈被证在架构、非超参。H 轮即来修此处。
        #   空间模式（H 轮 spatial_head=true）：1×1 卷积降到 grid_ch 通道 -> 池化到 pool_grid²
        #       -> 展平 = grid_ch × pool_grid²，取 = embed_dim（8×8×32=2048）。与 r8_norm 的 8×8
        #       网格同构、但为可学表征，空间排列被完整保留。
        if spatial_head:
            assert embed_dim % (pool_grid * pool_grid) == 0, (
                f"spatial_head 要求 embed_dim({embed_dim}) 能被 pool_grid²({pool_grid ** 2}) 整除")
            grid_ch = embed_dim // (pool_grid * pool_grid)
            self.pool = nn.Sequential(
                nn.Conv2d(aff_channels, aff_channels, 3, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(aff_channels, grid_ch, 1),      # 降通道：512 -> grid_ch
                nn.AdaptiveAvgPool2d(pool_grid),          # 保留 pool_grid×pool_grid 空间网格
                nn.Flatten(),
            )
            pool_out = embed_dim
        else:
            self.pool = nn.Sequential(
                nn.Conv2d(aff_channels, aff_channels, 3, padding=1),
                nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            pool_out = aff_channels
        # 投影头：LayerNorm -> Linear -> BatchNorm1d。LayerNorm 不是凑数：AFF 输出幅值可达
        # ~7e3，直接进 Linear 时 512 项求和在 fp16 下会溢出成 inf（baseline 踩过这个坑）。
        # 末尾的 BatchNorm1d 是为了打破初始对称性：实测初始网络对随机噪声也输出余弦 1.0 的
        # 嵌入（所有样本挤在同一方向），head 前各段的批内 std 极低；BN 按批逐维去均值，把这种
        # 近乎常数的输出重新拉开，是对比学习里项目头的标准做法（SimCLR）。
        head_in = pool_out + len(self.lateral) * aff_channels
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def _block(self, blk, x):
        """按需做梯度检查点：训练态不存该块的中间激活、反向时重算，省下约一半激活显存
        （代价约 +30% 算力）。评估态在 no_grad 下直接跑，不走检查点。"""
        if self.grad_checkpoint and self.training:
            return checkpoint.checkpoint(blk, x, use_reentrant=False)
        return blk(x)

    def _fusion_tail(self, *feats):
        """AFF + 池化段。抽成独立函数只为对它做梯度检查点：这段是 256px 全规格下的显存大头
        —— AFF 把两个尺度的特征都插值到 256² 并调整到 512 通道，单个张量就有 537MB
        （batch 4），通道拼接后 (B,1024,256,256) 更是 1GB+，且这些中间量原本要留到 backward；
        检查点化后只在反向重算时短暂出现，峰值可再降约 2-3GB。"""
        return self.pool(self.aff(list(feats)))

    def forward(self, x):
        # 整段前向强制 fp32。根因：256px 全规格 + 真实退化数据下，多分支残差块的中间激活幅值
        # 可达 O(1e3)，一旦某处卷积累加在 fp16（上限 65504）下溢出成 inf，后面所有环节（1×1 卷积、
        # 池化求平均、LayerNorm）都会把 inf 变成 NaN，整批损失报废。
        # 实测只把 AFF+head 升到 fp32 不够（第 40~60 步仍出 z=nan、参数正常），必须整段走 fp32。
        # 代价可接受：本模型只有 12.81M 参数、无解码器/检测头，比 baseline 的 HashVAE 轻得多。
        with torch.amp.autocast("cuda", enabled=False):
            x = self.conv0(x.float())
            feats, late = [], []
            for s, mb in enumerate(self.scales):
                x = self._block(mb, x)
                feats.append(x)
                if len(self.lateral):
                    late.append(self.lateral[s](x))     # 降采样前的特征图 -> 一条多尺度 lateral
                if s < len(self.downsample):
                    x = self._block(self.downsample[s], x)
            if self.grad_checkpoint and self.training:
                pooled = checkpoint.checkpoint(self._fusion_tail, *feats, use_reentrant=False)
            else:
                pooled = self._fusion_tail(*feats)
            z = self.head(torch.cat([pooled] + late, dim=1) if late else pooled)
        return z

    @staticmethod
    def build_from_config(cfg: dict) -> "SourceEncoder":
        return SourceEncoder(
            in_channels=3,
            base_channels=int(cfg.get("base_channels", 64)),
            num_branches=int(cfg.get("num_branches", 5)),
            num_rb=int(cfg.get("num_rb", 5)),
            num_scales=int(cfg.get("num_scales", 2)),
            aff_channels=int(cfg.get("aff_channels", 512)),
            embed_dim=int(cfg.get("embed_dim", cfg.get("hash_dim", 2048))),
            grad_checkpoint=bool(cfg.get("grad_checkpoint", True)),
            use_multiscale_skip=bool(cfg.get("use_multiscale_skip", False)),
            spatial_head=bool(cfg.get("spatial_head", False)),
            pool_grid=int(cfg.get("pool_grid", 8)),
        )
