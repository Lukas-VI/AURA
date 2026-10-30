# AURA 模型结构设计图（mermaid）

> 用途：把 `aura/` 自建单编码器的**实际结构**画成图，供讨论 / 交接 / 写论文时参考。
> 维护约定：**代码是唯一事实来源**，本文件随结构改动同步更新（改动清单见文末「同步记录」）。
> 对应代码：[encoder.py](file:///d:/py/AURA/aura/encoder.py)、[data.py](file:///d:/py/AURA/aura/data.py)、[geo.py](file:///d:/py/AURA/aura/geo.py)、[eval.py](file:///d:/py/AURA/aura/eval.py)、[train.py](file:///d:/py/AURA/aura/train.py)
> 日期：**2026-10-20（F–K2 轮：去 GAP 空间头 + 局部描述子头 + 可微几何支路 + 方案 B 两阶段重排器）**

***

## 1. 端到端总览（数据 → 编码 → 损失 / 检索）

```mermaid
flowchart TB
    A["FF++ C23 抽帧<br/>D:/ffpp_frames"] --> I["build_ffpp_index<br/>train 300 / val 100 / test 200 个真实视频"]
    I --> DS["ContrastiveDataset<br/>image_size=256, label = label_key（identity 源视频 X / source_id 具体源帧）"]

    DS -->|训练视图 train=True| TR["degrade：随机退化<br/>水平翻转 / 下采样 / JPEG / 模糊 / 噪声 / 亮度"]
    TR --> GA{"geom_aug?（F/K 轮开）"}
    GA -->|开| G1["随机几何增强<br/>旋转 / 缩放 / 平移（含掩膜同步）"]
    GA -->|关| ER
    G1 --> ER
    ER["erase_face：人脸区域随机擦除<br/>erase_prob（H/I/J 轮为 0.0）"] --> ENC["SourceEncoder（见 §2）"]

    DS -->|评估视图 train=False| CL["仅 resize + cvtColor<br/>不加退化 = clean 档"]
    CL --> ENC

    ENC --> Z["z ∈ R^2048（未归一化）"]
    Z --> LOSS["训练：SupCon + 1.0 × VICReg 方差铰链<br/>XBM 队列 4096（常驻 CPU）<br/>可选：低层蒸馏（G 轮）/ 可微几何一致性（K 轮）"]
    Z --> RET["检索（方案 B，两阶段）：<br/>初筛 top-K → 可微几何模块重排"]
    RET --> MET["帧级命中 = source_id<br/>视频级命中 = identity 前缀 X"]
```

***

## 2. SourceEncoder 前向（全量规格：256px / M=5 / D=5 / 64ch / aff512 / embed2048）

### 2.1 主干 → 池化段（两种模式）→ 全局头

```mermaid
flowchart TB
    X["x : B×3×256×256"] --> C0["conv0：Conv3×3(p1) → 64ch"]
    C0 --> S0["scale0：MultiBranchBlock(64, M=5, D=5)<br/>→ f0 : B×64×256×256"]
    S0 --> D0["downsample0：Conv3×3(stride2, p1) → 128ch<br/>→ B×128×128×128"]
    D0 --> S1["scale1：MultiBranchBlock(128, M=5, D=5)<br/>→ f1 : B×128×128×128"]

    S0 --> AFF
    S1 --> AFF["AFF：各尺度 1×1 调到 512ch → 双线性插值对齐到 256²<br/>→ 拼接(1024) → 1×1 注意力(1024→2)+Sigmoid → 加权和<br/>→ B×512×256×256"]

    AFF --> MODE{"spatial_head?"}
    MODE -->|false：A–G 轮 GAP 模式| POOL1["pool：Conv3×3(p1) → LeakyReLU → GAP(1×1) → Flatten<br/>→ B×512"]
    MODE -->|true：H/I/J/K 轮 空间模式| POOL2["pool：Conv3×3(p1) → LeakyReLU → Conv1×1(512→32)<br/>→ AdaptiveAvgPool2d(8×8) → Flatten<br/>→ B×2048（8×8×32，保留空间网格）"]

    POOL1 --> HEAD["head：LayerNorm(→) → Linear(→2048) → BatchNorm1d(2048)"]
    POOL2 --> HEAD
    HEAD --> OUT["z : B×2048（未归一化；归一化在损失 / 评估侧做）"]
```

> 注①：`use_multiscale_skip=true` 时每条 scale 在**降采样前**生成一条 `GAP → Linear(·→512)` 的 lateral，concat 进 head（head 输入 512→1536，参数 12.82M→15.02M）；当前各轮**始终关闭**，且已发现其 `AdaptiveAvgPool2d(1)` 不含空间相位 ⇒ 结构上无法修复帧级（见 HANDOFF §3.9 第 3 条）。
> 注②：`spatial_head=true` 只对 head 前的**池化段**做空间保留，head 本身不变；`embed_dim(2048)` 必须能被 `pool_grid²(64)` 整除。

### 2.2 局部描述子支路（`local_head`，K 轮新增；`forward_local`）

```mermaid
flowchart TB
    LX["x : B×3×256×256"] --> LC0["conv0 → 64ch"]
    LC0 --> LS0["scale0：MultiBranchBlock(64)<br/>→ B×64×256×256"]
    LS0 --> LH["local_head：Conv3×3(s2) → LReLU → Conv3×3(s2) → LReLU<br/>→ Conv3×3(s2) → LReLU → Conv1×1(32→local_dim=32)<br/>→ B×32×32×32"]
    LH --> LN["逐位置通道 L2 归一化"]
    LN --> LF["F : B×32×32×32（256px / 8 = 32 网格，对齐几何模块）"]
```

> 关键设计：`forward_local` **只跑到共享主干早期（conv0→scales[0]）**，不经 AFF/pool/head（那是 256px 下显存与算力的大头），但沿途参数与全局检索路径**共享** ⇒ 几何一致性损失的梯度能回传到 conv0/scales[0]，从而塑形检索表征。这正是 K 轮"几何损失进网络"与独立外挂重排器的区别（也是它失败的原因，见 HANDOFF §3.10 结论 5）。
> 独立外挂版不走主干：`LocalDescriptor`（`dim=32, width=32, depth=3`）**自己做稠密自监督 InfoNCE**（`_train_local_desc.py`），是方案 B 里表现最好的特征源（见 §7）。

***

## 3. MultiBranchBlock（主干的基本单元，M=5 分支 × D=5 残差块）

```mermaid
flowchart TB
    X["输入 x（B×C×H×W）"] --> P["分支串行计算 m = 0..M-1：<br/>m=0：out₀ = RB×D(x)<br/>m>0：outₘ = RB×D(x + mean(out₀..outₘ₋₁))"]
    P --> W["softmax(fusion_weights[M]) 加权求和<br/>out = Σ wₘ · outₘ"]
    W --> R["长残差：out + x"]
    R --> O["输出（尺寸 / 通道不变）"]
```

`ResidualBlock`（D 个串联）：`Conv3×3 → LeakyReLU → Conv3×3 → 残差相加 → LeakyReLU`。

***

## 4. AttentionFeatureFusion（AFF，把两个尺度对齐后按注意力加权）

```mermaid
flowchart TB
    F0["f0 : B×64×256×256"] --> A0["Conv1×1 → 512ch"]
    F1["f1 : B×128×128×128"] --> A1["Conv1×1 → 512ch"]
    A0 --> RS["双线性插值对齐到基准尺寸 256²"]
    A1 --> RS
    RS --> CC["通道拼接 → B×1024×256×256"]
    CC --> AT["注意力：Conv1×1(1024→2) + Sigmoid<br/>→ 权重 B×2×256×256"]
    AT --> SUM["按尺度加权求和 Σ wᵢ · aᵢ"]
    RS --> SUM
    SUM --> OUT["→ B×512×256×256"]
```

***

## 5. 损失

```mermaid
flowchart LR
    Z["z : B×2048"] --> N["L2 归一化"]
    N --> SC["SupCon：正 = 同 label_key，<br/>负 = 批内其他源 + XBM 队列 4096"]
    Z --> V["variance_loss：逐维 std 铰链（防嵌入坍缩）"]
    SC --> L["loss = loss_c + λ_vicreg · loss_v<br/>（+ λ_distill · loss_distill<br/>（+ λ_geo · loss_geo））"]
    V --> L

    Z -. G 轮 .-> DIS["低层蒸馏：投影头 2048→192<br/>对齐 r8_norm 教师（λ=0.5）"]
    DIS -.-> L
    FL["forward_local 特征 F"] -. K 轮 .-> GEO["可微几何一致性：<br/>正对 = 退化伪造 ↔ 同源干净真帧<br/>负对 = 批内错位异源真帧"]
    GEO -.-> L
```

- **SupCon（tau=0.2）+ 1.0 × variance_loss**：主线恒开。
- **低层蒸馏（G 轮）**：投影头 `2048→192` 对齐逐图标准化缩略图教师 `r8_norm`，λ=0.5；**结论：零增益**（e6 47.04 < A 轮 47.75）。
- **可微几何一致性（K 轮）**：正对 = 同 `source_id` 组的「干净真帧 ↔ 随机几何退化的伪造帧」，负对 = 批内错位异源真帧；铰链 margin=0.3，λ_geo=1.0，`temp=0.05`、`sigma=0.08`、`grid=32²`。**结论：双重证伪**（K 58.71 / K2 57.96 均低于 I-best 59.68）。

***

## 6. 退化基准（只作用于查询；图库保持干净且只编码一次）

### 6.1 光度六档（A–E 轮，`DEGRADATION_LEVELS`）

```mermaid
flowchart LR
    DB["图库：测试集真实帧 1600"] --> ENC1["encode → db_z"]
    Q["查询：测试集伪造帧 8000"] --> DG{"档位"}
    DG -->|clean| E1["原图"]
    DG -->|jpeg50 / jpeg20| E2["JPEG 重压缩"]
    DG -->|scale0.5| E3["0.5× 缩放再插回"]
    DG -->|crop0.5| E4["中心裁剪 50% 再插回"]
    DG -->|reencode| E5["0.75× → JPEG70 → 插回 → JPEG60"]
    E1 --> ENC2["encode → q_z（逐档重建）"]
    E2 --> ENC2
    E3 --> ENC2
    E4 --> ENC2
    E5 --> ENC2
    ENC2 --> RK["cosine / hamming Top-K → 帧级 & 视频级命中率"]
    ENC1 --> RK
```

### 6.2 几何七档（F–K2 轮，`GEOM_LEVELS`）

```mermaid
flowchart LR
    GQ["查询：伪造帧"] --> GD{"几何档（含 clean）"}
    GD -->|clean| G0["原图"]
    GD -->|crop0.8| G1["中心裁剪 0.8"]
    GD -->|crop0.7off| G2["0.7× 非中心裁剪"]
    GD -->|zoom1.15| G3["放大 1.15×"]
    GD -->|rot5 / rot10| G4["旋转 ±5° / ±10°"]
    GD -->|shift8| G5["平移 8%"]
    G0 --> GE["encode → q_z"]
    G1 --> GE
    G2 --> GE
    G3 --> GE
    G4 --> GE
    G5 --> GE
    GE --> GK["cosine Top-K → 帧级命中率"]
    DB2["图库：1600 真实帧（干净）"] --> GE2["encode → db_z"]
    GE2 --> GK
```

> 脚本：光度六档走 [eval.py `evaluate_degradation`](file:///d:/py/AURA/aura/eval.py)；几何七档走一次性脚本 [\_eval_geom_model.py](file:///d:/py/AURA/aura/runs/_eval_geom_model.py)（逐档重建查询嵌入）。

***

## 7. 方案 B：两阶段检索器（可微几何模块替代 SIFT+RANSAC）

```mermaid
flowchart LR
    Q["查询伪造帧"] --> P1["初筛：廉价描述子<br/>r8_norm ⊕ I-best 深度嵌入（逐查询 z-score 融合）"]
    DB["图库 1600 真实帧"] --> P1
    P1 --> TOPK["top-K 候选（K=100..800）"]
    TOPK --> GEO["候选内逐对打分：geo.py pair_score<br/>软互最近邻 → 软期望对应点<br/>→ 加权可微 DLT（lstsq, h33=1）<br/>→ 软内点占比"]
    Q -. 局部特征 .-> GEO
    GEO --> RANK["按几何分重排 → 帧级 Top-1"]
```

- 特征源两选：`local_head`（K 轮挂主干的局部头）/ `local`（**独立训练的 `LocalDescriptor`**）。
- `pair_score` 全链可微、**无 SIFT / 无 RANSAC**；`temp=0.05`、`sigma=0.08`、`grid=32²`（N=1024）。
- 读数（K=300，n_query=300）：`local` 均值 **66.52%** vs 经典 SIFT+RANSAC **67.57%**（−1.05pp，7 档 4 档反超）；`local_head` 63.91%。详见 HANDOFF §3.11。

***

## 8. 关键规格与工程决策（与图对照）

| 项 | 取值 | 说明 |
| --- | --- | --- |
| 输入 / 输出 | 256×256×3 → R^2048 | 整图输入，无前景/背景分离 |
| 主干 | M=5（分支）× D=5（残差块）× 2 尺度 | 沿用论文"全量规格"以保持可比性 |
| 通道 | 64ch（尺度0）→ 128ch（尺度1）→ aff 512 | |
| 嵌入头 | **两种模式**：GAP（A–G 轮，默认，输出 512）；**空间保留 `spatial_head`（H/I/J/K 轮主线）**：`Conv1×1(512→32) → AdaptiveAvgPool2d(8×8) → Flatten` = 8×8×32 = 2048 | 去 GAP 是自建线唯一的架构级跃升（+12pp）；GAP 的 65536:1 多对一满射是帧级瓶颈（HANDOFF §3.9） |
| 多尺度跳连 | `use_multiscale_skip`：每尺度降采样前特征图 → GAP → Linear(·→512) | 当前各轮**关闭**；已知其 `AdaptiveAvgPool2d(1)` 不含空间相位，无法修复帧级 |
| 局部描述子头 | `local_head`（K 轮）：共享 `conv0→scales[0]`，3 层 stride2 → (B,32,32,32)，逐位置 L2 归一化 | 供几何支路；梯度可回传共享主干。独立版见 `LocalDescriptor`（`_train_local_desc.py`） |
| 人脸擦除 | `erase_prob` 概率擦除人脸区（噪点/纯色/空洞/随机椭圆），阈值只在 (0.35,0.65) 随机 | BiSeNet 掩膜实测占画面约 **55%** 且近似二值（阈值抬到 0.7 面积仅 0.556→0.551），**强度只能靠 `erase_prob` 控**；H/I/J/K 轮为 **0.0** |
| head | LayerNorm → Linear → BatchNorm1d | LayerNorm 防 fp16 溢出根因；BN 打破初始对称性（防坍缩） |
| 前向精度 | 整段 **fp32**（`autocast(enabled=False)`） | fp16 下多分支激活幅值 O(1e3) 会溢出成 inf → NaN |
| 梯度检查点 | 可配；E/F/H/K 轮**开**（峰值 ~7.6GB）、I/J 轮**关**（22GB 富余换回 ~25–30% 算力） | 梯度数值等价，不改变训练动力学 |
| 监督粒度 | `label_key`：`source_id`（当前各轮，具体源帧 `X_f{帧号}`）/ `identity`（粗粒度备选） | 与帧级判据对齐；真正的难点是"同视频相邻帧"这种近乎相同的极难负样本 |
| batch / step | 2 源 × 2 张 = 4；800 步/轮 × 6 轮（K 轮 1 epoch 看方向） | 300 源 → 每轮只过约 22% 数据 |
| 损失 | SupCon（tau=0.2）+ 1.0 × variance_loss（+ 可选 蒸馏 λ0.5 / 几何一致性 λ1.0） | XBM 队列 4096 常驻 CPU |
| 几何模块超参 | `temp=0.05` / `sigma=0.08` / `margin=0.3` / grid 32²（N=1024） | sigma=0.08 > 网格单元 2/32=0.0625，软内点近乎失效（**未调优**） |
| 评测 | 图库 1600 真实帧 / 查询 8000 伪造帧 | 帧级 = source_id 命中；视频级 = 同源 X；光度六档 + 几何七档 |
| kill 硬线 | 帧级 Top-1 **> 60.05%**（零训练 `r8_norm`，2026-10-18 上移） | 自建线 best **59.68%（I e1）仍未过线** |

***

## 9. 同步记录

| 日期 | 同步内容 |
| --- | --- |
| 2026-10-17 | 新建；对齐 C 轮结构（identity + 人脸擦除 + 多尺度跳连，15.02M）与退化基准六档 |
| 2026-10-17 | C 轮（erase 0.5 + 跳连）帧级 Top-1 e1 32.23% → e2 13.06% 崩盘；改单变量隔离轮 `_erase_lite`（仅擦除、`erase_prob` 0.2、跳连关闭、12.82M）。补记掩膜近似二值、面积约 55% 的实测 |
| 2026-10-18 | `label_key` 改为可配置（identity / source_id），当前轮 `_fine` 用 `source_id`；规格表「监督粒度」「人脸擦除」「batch/step」行同步。结构本身未变 |
| 2026-10-20 | **F–K2 轮同步**：① §2 拆成「主干→池化段（GAP / **空间保留** 两模式）→全局头」与「局部描述子支路 `forward_local`」；② 新增 §7「方案 B 两阶段检索器（`geo.py pair_score`）」，插入原 §7→**§8 规格表**、§8→**§9 同步记录**；③ §5 损失补 蒸馏（G）/ 几何一致性（K）；④ §6 拆出**几何七档**；⑤ §1 总览加 `geom_aug` 与两阶段检索；⑥ 规格表补 `spatial_head` / `local_head` / 几何超参 / kill 硬线 60.05% 各行 |
