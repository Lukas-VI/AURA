# AURA 技术选型调研与后备方案（2026-10-24）

> 目的：针对"工程上难以涨点"的现状，完成一轮**外部技术调研 → 新技术选型 → 主选 + 多后备梯队**，
> 保证从**建模**层面模型性能可达标、可鲁棒。
> 前置事实：[AURA_HANDOFF.md](file:///d:/py/AURA/docs/AURA_HANDOFF.md)（A–K2 + 方案 B）与 10-21~23 未入档五实验
> （spatial_hn 58.99 / spatial_grid 60.05 追平硬线 / 8 选 1 协议 / selfgeo 65.58 / two_stage_selfgeo）。
> 结论速览：**主选方案 A = SSCD 拷贝检测配方微调 + ISC 冠军同款系统层（DBA/αQE/联合聚合）**；
> 备选 B = DINOv2+BoQ；C = XFeat/LightGlue 换局部特征源；D = 零训练纯系统层兜底；E = 取证指纹差异化。

***

## 0. "达标"的量化定义（先立靶再选型）

clean 帧级 Top-1 已被四方证据钉死在 ~60%（零训练上限探测 60.05；I 59.68 / hn 58.99 / grid 60.05 四条独立训练路径收敛 ±0.4pp；短路拼接零增益）。**因此"达标"绝不能定义成 clean 反超——那是模板匹配的最优解区间，任何建模都不该把算力花在那里。** 建模侧的达标定义如下（L1 只求持平，L2~L5 才是建模买点的地方）：

| 线 | 指标 | 达标阈值 | 现状最好 | 依据 |
| --- | --- | --- | --- | --- |
| **L1 硬线** | clean 帧级 T1（1600 库 / 8000 查询） | ≥ 60.05%（持平即可） | grid 60.05（已追平） | §3.6 / §3.9 饱和分析 |
| **L2 几何鲁棒线** | 单阶段全局描述子，几何 6 档（crop0.8/crop0.7off/zoom1.15/rot5/rot10/shift8）帧级 T1 均值 | 比 r8_norm 同档均值 **+8pp 以上** | I-best 仅 +3.2pp（~31.7% vs r8 ~28.5%） | openset_deg.json 12 档表内插 |
| **L3 系统线** | 两阶段 7 档均值（n=300 同口径） | > 经典 SIFT+RANSAC **67.57** | 可微几何 66.52（−1.1pp） | §3.11 / §3.12 |
| **L4 精细线** | 8 选 1 条件命中（随机 12.5%） | ≥ **68%** | geo32 外挂 66.25 / selfgeo32 65.58 | within_video_*.json |
| **L5 拒识线** | open-set 12 档拒识 AUC 均值 | ≥ r8 同档 **+2pp** | 与 r8 各有胜负、量级相当 | §3.14 退化档表 |

***

## 1. 调研结论：四个技术池

### 池 1 · 拷贝检测全局描述子（对位：初筛召回 + L2 鲁棒线）

**SSCD**（Meta，"A Self-Supervised Descriptor for Image Copy Detection"，Pizzi et al. 2022，arXiv 2202.10261）：

- **是什么**：专为"查询图是库内某图经过变换的拷贝"这一任务训练的对比描述子。训练配方 = **强增强（几何 + 光度 + 截图/overlay 类）生成正样本对 + SimCLR 式对比 + 熵正则 + 难负**——与我们任务结构**同构**（伪造帧 ↔ 源真实帧 = 共享背景的"拷贝对"）。
- **可得性**：官方仓库 [facebookresearch/sscd-copy-detection](https://github.com/facebookresearch/sscd-copy-detection)（**MIT**）。权重直接下载（dl.fbaipublicfiles.com，免 torch.hub）：`sscd_disc_mixup`（ResNet50, 512 维，推荐默认）、`sscd_disc_large`（ResNeXt101, 1024 维，精度最高档）。评估管线含 crop/rotation 扰动。
- **证据强度**：NeurIPS 2021 Image Similarity Challenge（1M 库 / 50K 查询 needle-in-haystack 拷贝检测）**全部冠军方案都建立在 SSL 拷贝检测描述子上**（冠军 matching track microAP **0.8329**）；获奖方案普遍叠加**局部-全局两级 + SIFT/RANSAC 重排 + 重增强**——与我们方案 B 的形态完全一致，等于**文献已验证我们这条流水线的骨架**，缺的只是那个被验证过的全局描述子本体。
- **对我们的意义**：这正是 §3.9-(2) 指出的"深度模型优化的是另一个目标"的对症解——SSCD 的目标函数**就是**"跨变换匹配拷贝"，与退化协议直接对齐；而 r8_norm 之类零训练描述子在几何档的崩塌（crop0.8 仅 20.59%）正是拷贝检测训练要消灭的失效模式。

**DINOv2 + 聚合头**（SALAD / BoQ / MixVPR，VPR 视觉位置识别线）：

- SALAD（Izquierdo & Civera, CVPR 2024, arXiv 2311.15937，[github](https://github.com/serizba/salad)）：DINOv2 + 最优传输聚合，单阶段 VPR SOTA。
- BoQ（Bag of Queries, ICCV 2024, [github](https://github.com/amaralibey/Bag-of-Queries)）：可学 query 向量 cross-attention 聚合 DINOv2 patch 特征，紧凑模型 Recall@1 领先 MixVPR/SALAD，权重开源（仓库连接超时未核 license，实施前核对）。
- **诚实警告**：我们自己冻住的 DINOv2 ViT-S/14 在 clean 上只有 57.05%（低于 tiny32 58.80）——**冻结通用 SSL 不是灵药**；这条池的价值在"**微调 + 任务化聚合头**"后的视角/光度剧变鲁棒性（VPR 文献的核心结论），定位为备选 B。

### 池 2 · 现代局部特征与匹配器（对位：重排器 + L4 精细线）

- **XFeat**（Potje et al., CVPR 2024, arXiv 2404.19174, [github](https://github.com/verlab/accelerated_features)）：**Apache-2.0**，torch.hub 一行加载，64 维紧描述子，~0.26M 参数，GPU 150+ FPS（VGA 单张）/ 批量 1400 FPS，CPU 快于 SIFT；含稀疏 + **半稠密**两种模式与 LighterGlue（比 LightGlue 快 3×，MegaDepth-1500 AUC@10 0.610 vs SuperPoint+LightGlue 0.633——用 1/5 算力换 96% 精度）。**2080Ti 上的理想重排前端**。
- **LightGlue**（2023，代码 Apache-2.0；注意 SuperPoint 官方权重非商用，**避开 SuperPoint 权重**，配 XFeat/ALIKED 用）。
- **OmniGlue**（Jiang et al., CVPR 2024, arXiv 2405.12979, [google-research/omniglue](https://github.com/google-research/omniglue)，Apache-2.0）：DINOv2 引导的泛化匹配器，OOD（未见域）上超 SuperGlue/LightGlue——**跨数据集/跨质量轴的重武器**。
- **与我们的关系**：现有可微几何模块（软互近邻 → 可微 DLT → 软内点）**保留不动**，只把特征前端从"InfoNCE LocalDescriptor / SIFT"换成 XFeat——"全链无 SIFT/RANSAC"叙事不变，且 8 选 1 的"细网格 + 对应"杠杆（8×8→32×32 单调 +8.5pp）有更高分辨率的现成实现。

### 池 3 · 检索系统层（零训练涨点，对位：L3 系统线 + 召回工程）

ISC 2021 冠军方案的公共形态 + 经典检索文献：

- **DBA（database augmentation，图库多视角索引）**：图库每帧存 K 个裁剪/缩放视角，查询对多视角取 max——直击 `crop0.7off` recall@20 = 73.8% 的封顶（经典出处 Gordo et al. 2017 DBAα）。
- **αQE（query expansion）**：查询用其 top 检索加权扩展再查（Radenović et al. TPAMI 2018 αQE）。
- **局部-全局两级 + 重排**：获奖方案标配（第 3 名 matching 即 SIFT+RANSAC 重排）。
- **联合 8×8 分配聚合**（我们自己的协议层增量）：同一伪造视频 8 条查询共享答案结构（帧号对齐），先视频级定位再帧级分配（匈牙利/软分配）——HashVAE（每视频 1 帧）与 ViTHash（无帧级）都没有这层。

### 池 4 · 取证专用信号（差异化备选，对位：跨质量轴 + related work 卡位）

- **源相机指纹溯源**：Deepfake Video Traceability via Source Camera Fingerprints（IET 2025）——PRNU 类信号做溯源，与我们"背景=指纹"假设互补；**PRNU 探针**（原 §4.2 #3 从未做）可作为可插拔第三路特征，1 天成本。
- **Attribution as Retrieval**（CVPR 2026）：生成指纹 + "归因即检索"框架——**论文 related work 的最近邻**，必须在引言里正面对比（我们是"源内容检索"，他们是"生成模型归因"）。
- FingerNet 类频域指纹（生成模型归因线）：不同任务，只引不追。

***

## 2. 主选方案 A：SSCD 配方微调 + 系统层组合

### 2.1 A-1 · 全局描述子换血（训练侧唯一主注）

**做法**：`sscd_disc_mixup`（ResNet50, 512 维）为初始化，在我们的数据上按**拷贝检测配方**微调：

1. **正样本对 = 同一 source_id**（真实帧 + 它的 5 种攻击版），**两边独立施加强增强**——随机裁剪（scale 0.3~1.0）、旋转（±30° 连续 + 90° 倍数档）、平移、缩放、JPEG(20~90)、模糊、噪声、亮度/对比度、降采样再回放（模拟"截图"档）。
2. SupCon（τ=0.2）+ SSCD 原版熵正则（或保留 VICReg 方差项）+ XBM 4096 队列 + **EMA（0.999~0.9995，双曲线都记）**。
3. lr 1e-5~2e-5（微调预训练权重），warmup 0.2，800 步/轮 × 6 轮；ResNet50 512 维在 22GB 上 batch 32+ 可行。
4. **Windows 对策**：AugLy 在 Windows 不可靠（字体/emoji 依赖）→ 用 kornia/imgaug 等价实现；`num_workers=0` 是吞吐瓶颈 → **增强搬到 GPU（kornia 批量做）或预解码缓存张量**，否则 IO 拖死训练。

**为什么它不会重蹈 F 轮覆辙**（关键论证，审稿人也会问）：

| | F 轮（失败） | A-1（本方案） |
| --- | --- | --- |
| 标签 | identity（同视频 8 帧互为正） | source_id（同帧的拷贝对互为正） |
| 增强的角色 | 对**不同样本各自独立**施加，要求模型对变换不变 → 与"帧级判据活在被抹掉的那层"直接对撞（§3.9-4） | 增强只发生在**拷贝对内部**，教会的是"跨变换匹配同一帧"——**不触碰**"同视频相邻帧要分开"的判据 |
| 初始化 | 随机 | 拷贝检测预训练权重（该配方已被 ISC 冠军验证） |

**验收门**：

- **G0 冻结探针（先跑，半天，零训练）**：`sscd_disc_mixup` 冻结权重直接过我们 12 档协议 + 8 选 1，与 r8 / I-best 同表。
  - 若几何 6 档均值已 ≥ r8 + 5pp ⇒ 方向直接确认，微调只做增益；
  - 若 clean 冻结读数 50~60 且几何档显著强于 I-best ⇒ 微调立项；
  - 若全面劣于 r8 ⇒ A-1 降级，切备选 B。
- **G1 微调验收 = L2**（几何 6 档 +8pp）；L1 持平即可，不作为验收项。

### 2.2 A-2 · 系统层（与 A-1 正交，纯推理，可与 G0 同日开工）

1. **DBA 多视角图库索引**（K=4~8 随机裁剪/缩放视角）——目标 `crop0.7off` recall@20 73.8% → 90%+；
2. **αQE**（一轮扩展，α≈0.3~0.5 起步）；
3. **联合 8×8 分配聚合**（新协议层：单帧层/聚合层两档指标，对 r8 / SIFT / SSCD **同样施加**，公平对照）；
4. **crop0.5 档口径修正**为随机偏移裁剪（现口径裁掉全部背景，测的是任务不可行性）。

**验收门 G2 = L3**：两阶段 7 档均值 > 67.57（经典 SIFT 同口径）。叠加 A-1 后目标 69~72（clean 档 SIFT 71.67 是锚点）。

### 2.3 时间与算力预算（2080Ti 22GB）

| 步骤 | 内容 | 成本 |
| --- | --- | --- |
| D0 | 四个冻结探针（SSCD×2 / DINOv2-S+BoQ / XFeat 半稠密接现有 geo 模块）过 12 档 + 8 选 1 | 半天，纯推理 |
| D1~D3 | A-1 微调（ResNet50）+ A-2 系统层并行 | 2~3 天（每轮 ~1.5h×6 epoch + 评估） |
| D4~D5 | 全表：12 档帧级 / 8 选 1 / open-set / 两阶段 / L1~L5 验收 | 1 天 |

***

## 3. 备选梯队（触发条件 → 动作）

| 梯队 | 内容 | 触发条件（何时切换） | 预期 |
| --- | --- | --- | --- |
| **B** | DINOv2(+registers) ViT-S 或 B/14 + **BoQ/SALAD 头** + 与 A-1 完全相同的增强配方微调（ViT-B 用 LoRA + 梯度检查点） | G0 中 SSCD 冻结全面劣于 r8；或 A-1 微调两轮不过 G1 | VPR 文献证明该形态对视角剧变 SOTA；BoQ 头参数量小、22GB 可驯 |
| **C** | 重排/精细战换血：XFeat(+LighterGlue) 或 ALIKED+LightGlue 作为局部特征前端，接**保留不动**的可微 DLT；OmniGlue 作跨数据集重武器 | 8 选 1 停在 <66；或两阶段与 SIFT 缺口 >2pp 且缺口在重排端（cond@1 落后）而非召回端 | XFeat 半稠密 + 32→64 网格，L4 目标 68+；"无 SIFT/RANSAC"叙事不变 |
| **D** | **零训练兜底**：SSCD 冻结 + DBA + αQE + 联合聚合 + XFeat 重排，全程不训练 | 训练预算被砍，或 A/B 微调全部失败 | 目标仅 L3（系统线）达标——保证论文"系统表"无论如何能超经典管线 |
| **E** | 取证信号差异化：PRNU 第三路特征探针（1 天）；频域指纹；跨质量轴（c0/raw 或不对称重编码）立项 | 跨质量协议轴上零训练与深度全面同跌（无交叉） | PRNU 对重压缩/降质天然稳健，是跨质量交叉的最可能来源；也是与 CVPR 2026 Attribution-as-Retrieval 划界的实证材料 |

**梯队组合律**：A-1 失败 ≠ 全盘失败——B 与 A-2/C/D 全部正交可叠加；A-2 与 C 无论训练侧成败都值得做（纯推理、无风险）。

***

## 4. 与既有负结果的相容性核对（防重蹈覆辙）

| 既有负结果 | 与本方案关系 |
| --- | --- |
| F 轮：几何增强反效果（clean −8.65pp） | 已排除同构——见 §2.1 对照表（identity 标签 + 独立增强 ≠ source_id + 拷贝对内增强） |
| K / K2 / 探索 2：几何目标进损失三方证伪 | 新方案**不动损失侧几何**；几何只在检索期当打分器（C 只换打分器的特征前端） |
| G 轮：低层蒸馏零增益 | 蒸馏对齐"输出"，不如直接换成**被验证的配方本身**（SSCD 初始化） |
| hn / grid：训练侧推不动 clean | 接受饱和结论，clean 只验收 L1 持平；主战场移到 L2~L5 |
| σ/temp 网格扫描 | 重排器参数侧已关死；新方案不改 σ/temp |
| 冻结 DINOv2 ViT-S clean 仅 57.05 | 已吸收为 B 桌上的诚实警告：通用 SSL 冻结不是灵药，必须任务化微调 + 聚合头 |

***

## 5. 风险清单

1. **离线/下载**：SSCD 权重（dl.fbaipublicfiles.com）、XFeat（torch.hub）、BoQ（GitHub release）需一次性联网；历史上已从 Kaggle 拉过 FF++，可行。下载后全程离线。
2. **Windows**：AugLy 装不上 → kornia/imgaug 等价实现；`num_workers=0` → GPU 端增强或预解码缓存（不做这条，D1~D3 的估时会翻倍）。
3. **显存**：ResNet50 微调无压力；ResNeXt101 1024 维（sscd_disc_large）与 ViT-B/14 需梯度检查点/LoRA，列为 B 梯队的规格选项。
4. **License**（学术发表全部可用）：SSCD **MIT**、XFeat **Apache-2.0**、LightGlue 代码 Apache-2.0（**SuperPoint 权重非商用，避开**）、OmniGlue Apache-2.0、DINOv2 Apache-2.0、BoQ 待核（仓库超时未确认，实施前核对）。
5. **预期管理**：clean 不会大涨（饱和是四方证据钉死的事实）；验收主体在 L2/L3/L4/L5。若 G0 探针 SSCD 在几何档也没有显著优势，说明"学习 buys 鲁棒性"在我们协议上也不成立——那也是重要负结果，直接强化路线 B 的分析论文立论，届时只做 C/D 两梯队保系统表。

***

## 6. 引用与链接

- SSCD：Pizzi et al., [A Self-Supervised Descriptor for Image Copy Detection](https://arxiv.org/abs/2202.10261)（2022）｜[facebookresearch/sscd-copy-detection](https://github.com/facebookresearch/sscd-copy-detection)（MIT）
- ISC/DISC21：Douze et al., [The 2021 Image Similarity Dataset and Challenge](https://arxiv.org/abs/2106.09672)｜Papakipos et al., [Results and findings of the 2021 ISC](https://proceedings.mlr.press/v176/papakipos22a/papakipos22a.pdf)｜[DrivenData 冠军解读](https://drivendata.co/blog/image-similarity-winners)（microAP 0.8329；SSL 是全部冠军公共组件）｜[获奖方案开源仓库](https://github.com/drivendataorg/image-similarity-challenge)
- SALAD：Izquierdo & Civera, [Optimal Transport Aggregation for VPR](https://arxiv.org/abs/2311.15937)（CVPR 2024）｜[serizba/salad](https://github.com/serizba/salad)
- BoQ：Amaralibey et al., [Bag of Queries](https://github.com/amaralibey/Bag-of-Queries)（ICCV 2024）
- XFeat：Potje et al., [XFeat: Accelerated Features](https://arxiv.org/abs/2404.19174)（CVPR 2024）｜[verlab/accelerated_features](https://github.com/verlab/accelerated_features)（Apache-2.0）
- OmniGlue：Jiang et al., [OmniGlue: Generalizable Feature Matching](https://arxiv.org/abs/2405.12979)（CVPR 2024）｜[google-research/omniglue](https://github.com/google-research/omniglue)
- 溯源近邻：[Deepfake Video Traceability via Source Camera Fingerprints](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/bme2/5687970)（IET 2025）｜[Attribution as Retrieval](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Attribution_as_Retrieval_Model-Agnostic_AI-Generated_Image_Attribution_CVPR_2026_paper.pdf)（CVPR 2026，related work 必引）
- QE/DBA 经典：Radenović et al. TPAMI 2018（αQE）；Gordo et al. 2017（DBAα）——凭记忆引用，成文前核对原文页码。

> 本文档只做选型，不动代码；实施顺序见 §2.3，第一步是 D0 冻结探针（半天、零训练、直接裁决 A/B/C 起点与 L2 可行性）。
