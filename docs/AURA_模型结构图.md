# AURA 模型结构设计图（mermaid）

> 用途：把 `aura/` 自建单编码器的**实际结构**画成图，供讨论 / 交接 / 写论文时参考。
> 维护约定：**代码是唯一事实来源**，本文件随结构改动同步更新（改动清单见文末「同步记录」）。
> 对应代码：[encoder.py](file:///d:/py/AURA/aura/encoder.py)、[data.py](file:///d:/py/AURA/aura/data.py)、[eval.py](file:///d:/py/AURA/aura/eval.py)、[train.py](file:///d:/py/AURA/aura/train.py)
> 日期：2026-10-17（C 轮：identity 监督 + 人脸掩膜擦除 + 多尺度跳连）

***

## 1. 端到端总览（数据 → 编码 → 损失 / 检索）

```mermaid
flowchart TB
    A["FF++ C23 抽帧<br/>D:/ffpp_frames"] --> I["build_ffpp_index<br/>train 300 / val 100 / test 200 个真实视频"]
    I --> DS["ContrastiveDataset<br/>image_size=256, label = identity（源视频 X）"]

    DS -->|训练视图 train=True| TR["degrade：随机退化<br/>水平翻转 / 下采样 / JPEG / 模糊 / 噪声 / 亮度"]
    TR --> ER["erase_face：人脸区域随机擦除<br/>erase_prob=0.5（噪点/纯色/空洞/随机形状）"]
    ER --> ENC["SourceEncoder（见 §2）"]

    DS -->|评估视图 train=False| CL["仅 resize + cvtColor<br/>不加退化 = clean 档"]
    CL --> ENC

    ENC --> Z["z ∈ R^2048（未归一化）"]
    Z --> LOSS["训练：SupCon + 1.0 × VICReg 方差铰链<br/>XBM 队列 4096（常驻 CPU）"]
    Z --> RET["检索：L2 归一化后 cosine Top-K<br/>或符号二值化后汉明 Top-K"]
    RET --> MET["帧级命中 = source_id<br/>视频级命中 = identity 前缀 X"]
```

***

## 2. SourceEncoder 前向（全量规格：256px / M=5 / D=5 / 64ch / aff512 / embed2048）

```mermaid
flowchart TB
    X["x : B×3×256×256"] --> C0["conv0：Conv3×3(p1) → 64ch"]
    C0 --> S0["scale0：MultiBranchBlock(64, M=5, D=5)<br/>→ f0 : B×64×256×256"]
    S0 --> L0["lateral0：GAP → Flatten → Linear(64→512) → LeakyReLU<br/>→ B×512"]
    S0 --> D0["downsample0：Conv3×3(stride2, p1) → 128ch<br/>→ B×128×128×128"]
    D0 --> S1["scale1：MultiBranchBlock(128, M=5, D=5)<br/>→ f1 : B×128×128×128"]
    S1 --> L1["lateral1：GAP → Flatten → Linear(128→512) → LeakyReLU<br/>→ B×512"]

    S0 --> AFF
    S1 --> AFF["AFF：各尺度 1×1 调到 512ch → 双线性插值对齐到 256²<br/>→ 拼接(1024) → 1×1 注意力(1024→2)+Sigmoid → 加权和<br/>→ B×512×256×256"]
    AFF --> POOL["pool：Conv3×3(p1) → LeakyReLU → GAP → Flatten<br/>→ B×512"]

    POOL --> CAT
    L0 --> CAT
    L1 --> CAT
    CAT["concat：[pooled(512) ‖ lateral0(512) ‖ lateral1(512)]<br/>→ B×1536"] --> HEAD["head：LayerNorm(1536) → Linear(1536→2048) → BatchNorm1d(2048)"]
    HEAD --> OUT["z : B×2048（未归一化；归一化在损失 / 评估侧做）"]
```

> 注：`use_multiscale_skip=false` 时 `lateral` 为空，`head` 退化为 `LayerNorm(512) → Linear(512→2048) → BatchNorm1d`，即无跳连版（12.81M 参数）；开启后 15.02M。

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

## 5. 损失与检索

```mermaid
flowchart LR
    Z["z : B×2048"] --> N["L2 归一化"]
    N --> SC["SupCon：正 = 同 identity，<br/>负 = 批内其他源 + XBM 队列 4096"]
    Z --> V["variance_loss：逐维 std 铰链（防嵌入坍缩）"]
    SC --> L["loss = loss_c + 1.0 · loss_v"]
    V --> L
```

***

## 6. 退化鲁棒基准（六档，只作用于查询；图库保持干净）

```mermaid
flowchart LR
    DB["图库：测试集真实帧 1600<br/>（干净，只编码一次）"] --> ENC1["encode → db_z"]
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

***

## 7. 关键规格与工程决策（与图对照）

| 项 | 取值 | 说明 |
| --- | --- | --- |
| 输入 / 输出 | 256×256×3 → R^2048 | 整图输入，无前景/背景分离 |
| 主干 | M=5（分支）× D=5（残差块）× 2 尺度 | 沿用论文"全量规格"以保持可比性 |
| 通道 | 64ch（尺度0）→ 128ch（尺度1）→ aff 512 | |
| 多尺度跳连 | 每尺度降采样前的特征图 → GAP → Linear(·→512) | 「半个 U-Net」编码器侧；用**可学 lateral**而非原始像素，避免背景逐像素泄漏。当前轮**关闭**（`use_multiscale_skip: false`，12.82M）；开启后 head 输入 512→1536、参数 15.02M |
| 人脸擦除 | `erase_prob` 概率擦除人脸区（噪点/纯色/空洞/随机椭圆），阈值只在 (0.35,0.65) 随机 | BiSeNet 掩膜实测占画面约 **55%** 且近似二值（阈值抬到 0.7 面积仅 0.556→0.551），**强度只能靠 `erase_prob` 控**；当前轮 0.2 |
| head | LayerNorm → Linear → BatchNorm1d | LayerNorm 防 fp16 溢出根因；BN 打破初始对称性（防坍缩） |
| 前向精度 | 整段 **fp32**（`autocast(enabled=False)`） | fp16 下多分支激活幅值 O(1e3) 会溢出成 inf → NaN |
| 梯度检查点 | 开（训练态，`use_reentrant=False`） | 反向峰值 17.0GB → 7.6GB |
| 监督粒度 | 可配置 `label_key`，当前 **`source_id`**（具体源帧 `X_f{帧号}`）；`identity` 为粗粒度备选 | 与帧级判据对齐：同视频其余 7 帧从正样本变负样本。实测两种粒度下**每锚点均有 1 个正样本**（batch=4 → 批内 2 个类别），故"正样本为 0"被排除；真正的难点是"同视频相邻帧"这种近乎相同的极难负样本，故本轮 lr 减半、warmup 加倍 |
| batch / step | 2 源 × 2 张 = 4；800 步/轮 × 6 轮 | 300 源 → 每轮只过约 22% 数据 |
| 损失 | SupCon（tau=0.2）+ 1.0 × variance_loss | XBM 队列 4096 常驻 CPU |
| 评测 | 图库 1600 真实帧 / 查询 8000 伪造帧 | 帧级 = source_id 命中；视频级 = 同源 X |

***

## 8. 同步记录

| 日期 | 同步内容 |
| --- | --- |
| 2026-10-17 | 新建；对齐 C 轮结构（identity + 人脸擦除 + 多尺度跳连，15.02M）与退化基准六档 |
| 2026-10-17 | C 轮（erase 0.5 + 跳连）帧级 Top-1 e1 32.23% → e2 13.06% 崩盘；改单变量隔离轮 `_erase_lite`（仅擦除、`erase_prob` 0.2、跳连关闭、12.82M）。补记掩膜近似二值、面积约 55% 的实测 |
| 2026-10-18 | `label_key` 改为可配置（identity / source_id），当前轮 `_fine` 用 `source_id`；规格表「监督粒度」「人脸擦除」「batch/step」行同步。结构本身未变 |
