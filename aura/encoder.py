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
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from baseline.models.encoder import AttentionFeatureFusion, MultiBranchBlock


class SourceEncoder(nn.Module):
    """整图 (B,3,H,W) -> 嵌入 (B, embed_dim)（未归一化，归一化由损失/评估侧做）。"""

    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_branches: int = 5, num_rb: int = 5, num_scales: int = 2,
                 aff_channels: int = 512, embed_dim: int = 2048,
                 grad_checkpoint: bool = True, use_multiscale_skip: bool = False,
                 spatial_head: bool = False, pool_grid: int = 8,
                 local_head: bool = False, local_dim: int = 32, local_width: int = 32):
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
        # 局部描述子头（K 轮新增）：与全局头【共享主干早期特征】（conv0 -> scales[0]），逐位置输出
        # L2 归一化描述子 (B, local_dim, H/8, W/8)（3 层 stride=2，256px -> 32×32 网格，对齐几何模块）。
        # 为什么必须挂在【共享主干】上：可微几何验证模块要的是"逐位置可为对应优化"的特征图，
        # 而几何一致性损失只有经 conv0/scales[0] 回传，才能塑形检索表征 —— 独立外挂的分支做不到。
        self.local_head = None
        if local_head:
            self.local_head = nn.Sequential(
                nn.Conv2d(channels_list[0], local_width, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
                nn.Conv2d(local_width, local_width, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
                nn.Conv2d(local_width, local_width, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
                nn.Conv2d(local_width, local_dim, 1),
            )
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

    def forward(self, x, return_pooled: bool = False):
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
        # return_pooled：把 head 之前的池化向量一并返回（spatial_head 模式下它就是
        # grid_ch × 8 × 8 的展平结果）。网格级对比损失要用【每个空间位置】的原始描述子，
        # 而 head 的 LayerNorm/Linear 会把位置混合掉，故必须在这一层取。
        if return_pooled:
            return z, pooled
        return z

    def forward_local(self, x):
        """只跑到共享主干早期（conv0 -> scales[0]）后接局部描述子头，返回逐位置 L2 归一化描述子。

        供 K 轮的可微几何验证模块使用：此行**不走 AFF/pool/head**（那是 256px 下显存与算力的大头），
        但沿途参数与全局检索路径共享 —— 几何一致性损失因此能回传到 conv0/scales[0]。
        """
        assert self.local_head is not None, "未启用 local_head（请在配置里设 local_head=true）"
        with torch.amp.autocast("cuda", enabled=False):
            x = self.conv0(x.float())
            x = self._block(self.scales[0], x)
            return F.normalize(self.local_head(x), dim=1)

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
            local_head=bool(cfg.get("local_head", False)),
            local_dim=int(cfg.get("local_dim", 32)),
            local_width=int(cfg.get("local_width", 32)),
        )


class LocalDescriptor(nn.Module):
    """轻量局部描述子网络（路线 1b）：整图 -> 逐位置 L2 归一化描述子 (B,D,H/8,W/8)。

    动机：路线 1 最小原型证实"可微几何验证模块"梯度可回传（Q2 过），但复用整图对比学习编码器
    的 feature map 做密集匹配时【匹配极度弥散】（锐度×N≈17~20），几何分数不判别（AUC≈0.39<随机）。
    根因是全局编码器只被训成"输出一个向量"，其特征在空间上平滑、不为逐位置对应优化。

    这里改为【自己训练】一个对应型局部描述子：用已知的随机几何变换做稠密自监督（InfoNCE），
    让"变换前后对应位置的描述子互相靠近、非对应位置互相远离"，再把它的特征图喂给同一个可微
    几何验证模块。默认三层 stride=2，共降采样 8×（256px -> 32×32 网格），与验证模块的坐标网格对齐。
    """

    def __init__(self, in_ch: int = 3, dim: int = 32, width: int = 32, depth: int = 3):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(depth):                          # 每层 stride=2 -> 总步长 2**depth（默认 8）
            layers += [nn.Conv2d(c, width, 3, stride=2, padding=1), nn.LeakyReLU(0.1)]
            c = width
        layers += [nn.Conv2d(c, dim, 1)]                # 1×1 投到描述子维
        self.net = nn.Sequential(*layers)
        self.grid_stride = 2 ** depth

    def forward(self, x):
        return F.normalize(self.net(x.float()), dim=1)  # 逐位置通道 L2 归一化


class ResNetLocalDescriptor(nn.Module):
    """ImageNet 预训练 ResNet18 浅层特征图 + 可训 1×1 投影，作为对应型局部描述子（A 方案）。

    动机：原 LocalDescriptor 只有 11K 参数、只靠合成仿射自监督，供给几何模块的位置判别力
    不够 —— clean 两阶段的条件正确率 cond_t1 卡在 69.7%（初筛 recall@20 已 99.5%，瓶颈全在
    重排这一步）。这里把"供给"换成百万参数预训练主干：

        conv1+maxpool（stride 4）-> layer1：64 通道 @ 64×64（256px 输入，细网格）
        layer2：128 通道 @ 32×32 -> 双线性上采样回 64×64（更大感受野的语义）
        拼接 192 通道 -> 1×1 conv 投到 dim -> 逐位置 L2 归一化

    主干默认冻结（只训 192→dim 的投影，约 6K 参数），BN 始终走 eval 的 ImageNet 统计量，
    避免 2400 张训练帧的批统计把预训练特征带偏。
    """

    def __init__(self, dim: int = 32, pretrained: bool = True, freeze_bn: bool = True,
                 train_trunk: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2 = r.layer1, r.layer2
        for p in self._trunk_modules():
            p.requires_grad_(bool(train_trunk))
        self.proj = nn.Conv2d(192, dim, 1)
        self.freeze_bn = bool(freeze_bn)
        self.grid_stride = 4                            # conv1(2) × maxpool(2)

    def _trunk_modules(self):
        return [self.conv1, self.bn1, self.layer1, self.layer2]

    def train(self, mode: bool = True):
        """主干 BN 永远保持 eval：预训练统计量比 2400 张训练帧的批统计更可靠。"""
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x):
        x = x.float()
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f1 = self.layer1(x)                                              # stride 4
        f2 = F.interpolate(self.layer2(f1), size=f1.shape[-2:],
                           mode="bilinear", align_corners=False)          # stride 8 -> 对齐
        return F.normalize(self.proj(torch.cat([f1, f2], dim=1)), dim=1)


def build_local_descriptor(backbone: str = "tiny", dim: int = 32, width: int = 32,
                           depth: int = 3, train_trunk: bool = False) -> nn.Module:
    """局部描述子工厂：tiny = 原 11K 自建网络；resnet18 = ImageNet 预训练主干 + 1×1 投影。"""
    if backbone in (None, "", "tiny"):
        return LocalDescriptor(dim=dim, width=width, depth=depth)
    if backbone == "resnet18":
        return ResNetLocalDescriptor(dim=dim, train_trunk=train_trunk)
    raise ValueError(f"未知局部描述子主干: {backbone}")
