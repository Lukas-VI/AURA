# AURA — 深度伪造主动检索（基线复现 + 自建线）

**AURA** (Attack-Unsupervised Retrieval of Authentic backgrounds)：给定一张深度伪造图，
检索出它对应的**真实源图**（利用背景作为来源指纹），同时输出真假判别。

分两个阶段，两条线各自独立：

- **阶段一 · 基线与协议（已冻结）**：复现论文 *Deep hashing framework for proactively deepfake
  retrieval and detection* (Liu et al., 2026) 的 HashVAE。论文未开源，本仓库按论文描述
  **重写了一版简化实现**，放在 `baseline/`，作为自建线的实验对照。**已冻结为"保真度样本"，不再改动。**
- **阶段二 · 自建 AURA 线（进行中）**：整图输入、单编码器、端到端对比学习的自建检索模型，放在 `aura/`。
  当前主线是**两阶段检索器**（廉价初筛 top-K → **可微几何模块**重排，全链无 SIFT/RANSAC）。
  现状与关键读数见 §6，逐轮读数 / 瓶颈 / 待决问题见 [docs/AURA_HANDOFF.md](file:///d:/py/AURA/docs/AURA_HANDOFF.md)。
  **路线定位（2026-10-20）**：放弃"方案 B 反超经典 SIFT+RANSAC"的目标，改走**benchmark / 分析论文框架** ——
  以【任务定义 + 退化协议 + open-set 拒识 + GAP 瓶颈分析 + 零训练上限探测 + 可微几何重排器（只持平、含负结果）】立篇，**不靠单一绝对数值**。

## 1. 目录结构

```
AURA/
├── README.md
├── requirements.txt
├── aura/                     # 自建 AURA 线（阶段二，当前主线）
│   ├── encoder.py            # SourceEncoder：多分支多尺度主干 + AFF + 嵌入头（含空间保留头 / 局部描述子头）
│   ├── data.py               # 数据集索引 / 前后景掩膜 / 退化（光度六档 + 几何七档）/ 几何增强
│   ├── geo.py                # 可微几何验证模块：软互最近邻 -> 可微 DLT -> 软内点（无 SIFT/RANSAC）
│   ├── losses.py             # SupCon + VICReg + 低层蒸馏 + 几何一致性
│   ├── eval.py               # 评测协议：1600 真实图库 / 8000 伪造查询；帧级 + 视频级 × 余弦 + 汉明
│   ├── train.py              # 训练入口（KILL_LINES 硬线判据在此）
│   ├── configs/              # 逐轮配置（aura_full / _geo / _distill / _spatial / _spatial_i / _spatial_j / _geo_k）
│   └── runs/                 # 权重 / history / 诊断 json / 一次性实验脚本（_eval_two_stage_geo.py 等）
└── baseline/                 # 简化版 HashVAE 基线（自包含包，已冻结）
    ├── __init__.py
    ├── configs/              # default.yaml 默认超参 + baseline.yaml / smoke.yaml 实验覆盖
    ├── data/                 # 数据集加载 / 前后景分离 / 伪造图合成 / 增强
    ├── models/               # 编码器 / 解码器 / 检测块 / HashVAE
    ├── losses/               # 重构 / 最小熵 / 比特平衡 / 四元组
    ├── eval/                 # 检索评估 / 检测评估
    ├── analysis/             # 诊断实验 + 可视化
    ├── scripts/              # 一键运行脚本（run_baseline.ps1）
    ├── train_baseline.py     # 训练（阶段1无监督 + 阶段2弱监督）
    ├── test_baseline.py      # 评估 Top-1/Top-5 检索 + AUC/ACC
    ├── utils.py              # 通用工具（配置 / 种子 / 设备）
    ├── checkpoints/          # 权重与 history（运行后生成）
    ├── logs/                 # tensorboard 日志（运行后生成）
    └── results/              # 指标 / 诊断 / 图（运行后生成）
└── paper/                    # LaTeX 论文（路线 B：benchmark / 分析框架；中文初稿）
    ├── main.tex              # 主控文件（ctexart + xelatex）
    ├── preamble.tex          # 宏包与自定义宏
    ├── refs.bib              # 参考文献
    ├── sections/             # 摘要 / §1 引言 / §2 相关工作 / §3 任务与协议 / §4 方法 / §5 实验 / §6 讨论
    ├── figures/              # 插图
    ├── build/                # 编译产物（main.pdf / *.aux / *.log 等，不入库）
    └── build.ps1             # 编译脚本（xelatex + bibtex，无 perl 依赖）
```

> `baseline/` 是自包含包：配置路径、权重、日志都以**包目录**为基准解析，
> 因此统一用 `python -m baseline.xxx` 在**项目根目录**下运行即可。
>
> `aura/` 的权重 / history / 日志一律落在 `aura/runs/`（配置里用绝对路径，避免落到 `baseline/` 下）；
> `.gitignore` 只跟踪 `runs/` 下的 `*.py`（一次性评估脚本），权重与 json 不入库。

## 2. 数据说明（FF++ C23 抽帧）

已用 **FaceForensics++ C23**（Kaggle `xdxd003/ff-c23`）抽帧后的**人脸帧**作为数据源，
默认目录 **`D:/ffpp_frames`**（由 `baseline/scripts/extract_ffpp_frames.py` 生成）：

```
D:/ffpp_frames/
├── manifest.csv                 # 抽帧清单：folder,stem,frame,frame_idx,video,...,image
├── original/000/f000.jpg ...    # 1000 个真实视频 × 8 帧（256x256 人脸裁剪）
├── Deepfakes/000_003/f000.jpg   # 5 种换脸算法，各 1000 个伪造视频 × 8 帧
├── Face2Face/000_003/...  FaceSwap/...  NeuralTextures/...  FaceShifter/...
└── DeepFakeDetection/01_02__scene__hash/...   # 命名不同，默认不使用（见下）
```

**伪造视频命名 `X_Y`（FF++ 官方约定）**：`X` = 目标视频（提供背景/头姿，即被换脸者所在的
真实视频），`Y` = 供体视频（提供人脸身份）。于是对一张伪造帧：

- 它的**背景来源真实帧** = `original/X` 的同序号帧 → 这就是检索要找回的**源图**（`source_id`）；
- 它的**人脸来源真实帧** = `original/Y` 的同序号帧 → 作为四元组的**负样本**（背景与身份都不同）。

**划分**：按**真实视频编号 X** 切 train/val/test（`num_train_videos/num_val_videos/
num_test_videos`）。同一视频的帧必然落在同一 split，伪造图与其源真实图也必然同 split，
检索评估不会因为切分而泄漏答案。

> 为什么换成真实伪造图：早期用“合成换脸（干净贴脸）”当伪造样本时，前景里没有任何可
> 泛化的伪造痕迹，检测块 DDB 学不到东西（详见 5.1）。FF++ 的伪造帧是真实换脸算法生成的，
> 带网络生成指纹/融合伪影，正是 DDB 依赖的信号。
>
> `DeepFakeDetection/` 的命名是 `01_02__scene__hash`，其 original 不在本仓库这套
> `000..999` 里，无法与 `original/X` 配对，故 `manipulations` 默认不含它。

## 3. 环境

- Python 3.x + PyTorch + torchvision + opencv + scikit-learn + umap-learn + matplotlib + tqdm + pyyaml + tensorboard

## 4. 使用

所有命令都在**项目根目录** `d:\py\AURA` 下执行：

```powershell
# 冒烟测试（小规模，确认能跑通）
E:\conda\envs\pytorch\python.exe -m baseline.train_baseline --config configs/baseline.yaml --limit 24 --epochs_stage1 1 --epochs_stage2 1 --batch_size 8

# 完整训练
E:\conda\envs\pytorch\python.exe -m baseline.train_baseline --config configs/baseline.yaml

# 评估 / 诊断 / 可视化
E:\conda\envs\pytorch\python.exe -m baseline.test_baseline --config configs/baseline.yaml
E:\conda\envs\pytorch\python.exe -m baseline.analysis.diagnose  --config configs/baseline.yaml
E:\conda\envs\pytorch\python.exe -m baseline.analysis.visualize --config configs/baseline.yaml

# 或者用一键脚本
powershell -ExecutionPolicy Bypass -File baseline/scripts/run_baseline.ps1 -Stage all
```

结果默认写入 `baseline/checkpoints/`（权重、history）、`baseline/logs/`（tensorboard）与
`baseline/results/`（指标、诊断、图）。

### 4.1 自建线（`aura/`）

同样在项目根目录执行，需带上 `PYTHONPATH=d:\py\AURA`；权重 / history / json 落在 `aura/runs/`：

```powershell
$env:CUDA_DEVICE_ORDER="PCI_BUS_ID"; $env:CUDA_VISIBLE_DEVICES="1"; $env:PYTHONPATH="d:\py\AURA"

# 训练（逐轮配置：_geo / _distill / _spatial / _spatial_i / _spatial_j / _geo_k）
E:\conda\envs\pytorch\python.exe -u -m aura.train --config aura/configs/aura_full_geo_k.yaml

# 七档几何退化评估（图库保持干净、只对查询退化）
E:\conda\envs\pytorch\python.exe -u aura/runs/_eval_geom_model.py --ckpt aura/runs/aura_full_spatial_i_best.pth

# 两阶段检索器：初筛 top-K -> 可微几何模块重排（全链无 SIFT/RANSAC）
E:\conda\envs\pytorch\python.exe -u aura/runs/_eval_two_stage_geo.py --features local --prefilter fuse

# 方案 B 参数敏感性扫描（sigma × temp 网格，纯推理不重训）
E:\conda\envs\pytorch\python.exe -u aura/runs/_scan_geo_params.py --n_query 300 --features local --ks 100,300 --temps 0.01,0.03,0.05 --sigmas 0.02,0.04,0.06,0.08

# 开集拒识协议（闭集/开集切分 + 拒识 AUC / TPR@FPR）
E:\conda\envs\pytorch\python.exe -u aura/runs/_eval_openset.py
```

> 一次性实验脚本（`_probe_*.py` / `_eval_*.py` / `_preheat_local_head.py` / `_train_local_desc*.py` / `_scan_geo_params.py` 等）
> 都在 `aura/runs/` 下，是唯一入库的 `.py`（见 §1 末注）。`num_workers=0`（Windows 必须）。

## 5. 与论文 (Liu et al. 2026) 的对齐与差异

本次已**按论文原文**修正模型的**结构**（前后景分离、共享编码器、DDB 只读前景哈希、
四元组前景/背景分项），关键修正如下：

1. **前后景分离编码（论文 3.2/3.3.2）**：编码器分别作用于前景 `x_f`(人脸) 与背景 `x_b`，
   且**共享参数**，分别得到前景哈希 `F_f,hash` 与背景哈希 `F_b,hash`；哈希码 `C = C_f + C_b`。
   （旧实现把整图编成单一哈希，前后景混在一起。）
2. **检测块 DDB 只读前景哈希（论文 3.4.1）**：`DDB` 输入是 `F_f,hash`，而非前后景拼接的完整哈希。
   （旧实现把混合哈希喂给检测块，真假样本被四元组压到一起，检测器无法区分。）
3. **四元组分前景项/背景项（论文式 15）**：
   - 前景项：`‖C_f(q) − C_f(q+1)‖² − ‖C_f(q) − C_f(q−)‖² + α`；
   - 背景项：`‖C_b(q) − C_b(q+2)‖² − ‖C_b(q) − C_b(q−)‖² + β`，`α=β=0.5`；
   - 距离用**平方 L2**（旧实现误用了开方距离）。
   - 正样本构造对齐论文语义：`q+1` 复用 q 的前景（换到别处背景，身份保持），
     `q+2` 复用 q 的背景（人脸换成另一身份）——保证“对齐的是对应模态”。

| 论文 (HashVAE) | 本项目实现 | 说明 |
| --- | --- | --- |
| 前后景分离 + 共享编码器，输出 `C_f`/`C_b`，`C=C_f+C_b` | 同论文 | 本次对齐 |
| DDB 只吃前景哈希 `F_f,hash`，稠密聚合线性层（式12） | 同论文 | 本次对齐 |
| 四元组前景项 + 背景项，平方 L2，α=β=0.5（式15） | 同论文 | 本次对齐 |
| 重构 L2 分别约束前景/背景（式9） | 同论文 | 本次对齐 |
| 最小熵比特损失作用于前后景拼接哈希（式10/11/16） | 同论文，另加“比特平衡”项 | 防哈希坍缩的工程增强 |
| 掩膜由**分割式**前后景划分（人脸检测 + 注意力擦除 top-n）生成 | **BiSeNet 19 类解析**：前景 = 类 1–18（一切人语义：脸/皮肤/头发/衣服/脖子），背景 = 类 0（纯场景）；掩膜随样本从 dataset 流入模型 | 默认 `fg_mask_source: 'parse'`；几何椭圆保留为 `'geometry'` 消融/兜底。**与论文属同类"分割式"机制 —— 是不同实现、不是贡献**；改用现成 BiSeNet 解析权重（零额外依赖），实测比椭圆显著提升检索（见 5.1，属**复现保真度修正**） |
| 式7：哈希特征经 softmax | 逐比特映射到 (0,1)（BN+Tanh） | 按字面在码维做 softmax 会挤成 one-hot → 哈希坍缩，故用逐比特映射 |
| M=5 分支 / D=5 RB / 64 通道；hash 2048 位 | 2 分支 / 2 RB / base 32 / aff 256；hash 128 位 | 早期受 4GB 显存（RTX 2050）限制的简化档；现已换到 22GB RTX 2080 Ti，容量放大列为下一步 |
| 数据集 FF++ / Celeb-DF / DFDC | FF++ C23 抽帧（D:/ffpp_frames） | 已换成真实伪造帧 |
| 标签 real=1 / fake=0 | real=0 / fake=1 | 仅符号约定差异 |
| 两阶段检索 Algorithm 1 | 检索评估用背景码 `C_b` 汉明距离 | 对应论文情形(1) 第一步；与 AURA“背景=来源指纹”假设一致 |
| faceswap/simswap 生成伪造图 | FF++ 真实伪造帧（q+2）；`q+1` 仍用自包含合成换脸 | q+1 只用于前景项，合成即可 |

> **界定提醒（2026-10-20 修订）**：上表"掩膜"一行**不是**我们的贡献 —— HashVAE 原论文用的是**分割式**前后景划分，本实现改用 BiSeNet 只是"同机制的不同实现"，属 **复现偏差（C 类）**，应诚实标注。其带来的 48.6× 跃迁（背景码 Top-1 0.0050 → 0.2428）应读作"**此前几何椭圆是欠保真的复现，改回分割式掩膜后保真度提升**"，而非新方法。
>
> 另：**`aura/` 自建线已开工并在推进**（阶段二，见 §6），但这条线的增量**仍只在任务 / 协议 / 溯源层，不在模型结构层** —— 编码器 / 掩膜 / 检测头都属对论文架构的沿用或复现。上一句 2026-10-16 的"`aura/` 仍是占位"已作废。详见 [docs/AURA_Retrieval方向备份提案与迭代计划.md §0.2](file:///d:/py/AURA/docs/AURA_Retrieval方向备份提案与迭代计划.md)。

### 5.1 现状与遗留问题（重要）

**一句话现状：掩膜改造（几何椭圆 → BiSeNet 解析）解锁了检索，但没有解锁检测。**

**检索（已可用）**：把"背景"从"人脸残片 + 耳麦/头巾/衣服等配饰 + 真场景"的混合物，
改为 BiSeNet 解析出的**纯场景（类 0）**，其余一切不变。同一测试协议（1600 真实图库 /
8000 伪造查询，随机基线 Top-1 = 0.0625%）下：

| 指标 | 几何椭圆（旧） | BiSeNet 解析（现） | 相对随机 |
| --- | --- | --- | --- |
| 背景码 Top-1 | 0.0050 | **0.2428** | 388× |
| 背景码 Top-5 | 0.0245 | **0.6443** | 206× |
| 完整码 Top-1 | 0.0076 | **0.3314** | 530× |
| 完整码 Top-5 | 0.0336 | **0.7984** | 256× |

即 `quad` 正常下降（2.73→1.49），且**背景码 Top-5 已达 64.4%** —— 背景确实是可用的
来源指纹，此前检索几乎失效的原因是**背景指纹被稀释**，而非假设错误。

**检测（仍未解锁）**：`cls` 三代实现（单哈希 / 几何椭圆 / BiSeNet 解析）**恒为 2·ln2 = ln4 ≈ 1.3863**
（`loss_c = BCE(pred_q,0) + BCE(pred_f,1)` 两项，DDB 输出恒 ≈0.5 时每项恰为 ln2，和 = ln4），
本轮检测 AUC = **0.5166 ≈ 随机**。诊断结论：

- 检测块 DDB 的**结构正确**——固定前景哈希、单独训 DDB，400 步内即可把损失从 0.75 降到 0.001；
- 绕开四元组/最小熵做**纯监督 BCE**，训练/测试 AUC 仍≈随机（train 0.482 / test 0.467）；
- 只有对**固定的一对**图训练时损失才下降（能记忆、不能泛化）。

归因（**已两次修正**）：
1. 早期把原因归为"合成换脸是干净贴脸、前景无伪影"，并建议换真实 Deepfake 数据集 —— **换完
   FF++ 真实伪造帧后 `cls` 依旧恒为 ln4，该归因作废**；
2. 随后怀疑是"目标冲突"（DDB 只读前景哈希，而前景要同时满足四元组前景项与最小熵，真假被压近）
   —— 但这只是假说，且与"码里压根没有信息"的修法**相反**。故做**零训练判别探针**
   （[signal\_probe.py](file:///d:/py/AURA/baseline/analysis/signal_probe.py)：冻结编码器，抽测试集
   9600 张的 `h_f`，按视频号分组切 train/test 训简单分类头）：

   | 特征 | 线性 train/test AUC | MLP train/test AUC |
   | --- | --- | --- |
   | `h_f`（DDB 的输入） | 0.5565 / **0.5105** | 0.8567 / **0.5227** |
   | `h_b`（对照） | 0.5244 / **0.5010** | 0.6175 / 0.4987 |

   **裁决：前景哈希码里没有可泛化的真假信息**（MLP 能拟合训练集 0.86 却测不出测试集 0.52）。

因此 `cls≡ln4` **不是**损失编排问题（加 detach / 改损失调度都治不了信息缺失，该方向已排除）；用户提出的显存红利（4GB→22GB）恰好是唯一对口的方向。

**下一步**：① 提容量 + 开大 batch（`base_channels` 32→64、`aff_channels` 256→512、`hash_dim` 128→2048，stage2 batch 2→16）；② 提输入分辨率（128px 原为 4GB OOM 的妥协，现可放宽到 256px，但属约束变更，需确认）；③ FF++ C23 的高频伪影被重压缩 + 下采样摧毁，可考虑更高质量的数据版本（c0/raw）。详见 [docs/AURA\_研究方向评估与实现方案.md](file:///d:/py/AURA/docs/AURA_研究方向评估与实现方案.md) §11.6。

## 6. 自建 AURA 线：现状与关键读数（2026-10-20）

### 6.1 协议（已冻结，两条线共用）

- **图库** = test split 的 **1600 张真实帧**（干净，只编码一次）；**查询** = test split 的 **8000 张伪造帧**。
- **帧级命中** = 精确 `source_id`（`X_f{帧号}`，不是"同视频即命中"）；**视频级命中** = 同源视频 `X`。
- 随机基线：帧级 Top-1 = 1/1600 = **0.0625%**，Top-5 = **0.3125%**。
- **撤销/退化档**：几何七档 `clean / crop0.8 / crop0.7off / zoom1.15 / rot5 / rot10 / shift8`（方案 B 主用）；
  另有一套光度六档 `clean / jpeg50 / jpeg20 / scale0.5 / crop0.5 / reencode`（早期）。**只对查询退化，图库保持干净。**
- **kill 硬线（2026-10-18 已上移）**：帧级 Top-1 > **60.05%**（零训练 `r8_norm`：8×8 逐图标准化缩略图，仅 192 维，同协议同图库）——不过则"训练出来的模型打不过零训练曲线"。旧线 58.80%（零训练 32×32 整图缩略图）已被取代。**自建线最好成绩 I-best 59.68% 仍未过此线。**

### 6.2 各轮帧级 Top-1（clean 档，余弦）

| 轮 | 唯一变量 | 帧 Top-1 | 备注 |
| --- | --- | --- | --- |
| A | identity 粗粒度 | 47.75%（e6） | 基线 |
| B | source_id 细粒度 | 38.92%（e6） | **不稳，已作废**（E 轮证明是超参问题） |
| C | 擦除 0.5 + 多尺度跳连 | 32.23% → 13.06% | **崩盘，已停** |
| D | 擦除 0.2（单变量） | 46.91%（e6） | 与 A 持平 ⇒ "人脸捷径"未被证实 |
| E | 稳定版细粒度 | 46.44%（e6） | 只买到稳定，没买到精度 |
| F | 几何增强 `geom_aug` | 37.81%（e6） | 反向：训练不变性与帧级判据冲突 |
| G | 低层蒸馏（r8_norm 教师） | 47.04%（e6） | 路线 A，零增益 |
| H | **去 GAP（空间保留头）** | **58.45%（e1）** | **+12pp 跃升**，但 e2/e3 回落到 53.20/52.21 |
| I | H + 稳训（lr↓↓、warmup↑） | **59.68%（e1 best）** | 越过所有前轮，但仍 < 现行硬线 60.05%；e2/e3 仍单调回落 |
| J | 冻结 backbone，只训 head | 55.46%（e1） | 证明回落来自 SupCon 侵蚀空间布局 |
| K | local_head + 几何损失进网络 | 58.71%（e1） | **未涨** |
| K2 | 同 K，从 preheat 权重起 | 57.96%（e1） | **再次未涨** ⇒ 几何损失进网络被双重证伪 |

> 结论：**GAP（head 前的全局平均池化，65536:1 多对一满射）就是帧级瓶颈**，去 GAP 一次跃升 +12pp；
> 但队列灌满后梯度会持续侵蚀空间布局，训得越久越掉。**几何一致性损失当辅助项训进网络无效**（K/K2）。

### 6.3 方案 B 终局：两阶段检索器（可微几何模块替代 SIFT+RANSAC）

口径：廉价初筛 top-K → 候选内用可微几何模块（[geo.py `pair_score`](file:///d:/py/AURA/aura/geo.py)：软互最近邻 → 可微 DLT → 软内点）
重排；`n_query=300`（与经典管线同查询子集）、初筛 = r8_norm 与 I-best 深度嵌入逐查询 z-score 融合、K=300、grid 32²。
脚本 [\_eval_two_stage_geo.py](file:///d:/py/AURA/aura/runs/_eval_two_stage_geo.py)。**全链无 SIFT/RANSAC。**

| 档位 | 初筛单独 T1 | local_head 纯几何 | **local（独立描述子）纯几何** | 经典 SIFT+RANSAC |
| --- | --- | --- | --- | --- |
| clean | 60.00 | 66.00 | 67.00 | 70.67 |
| crop0.8 | 25.00 | 64.67 | 69.00 | 68.00 |
| crop0.7off | 5.67 | 54.33 | 57.67 | 56.67 |
| zoom1.15 | 38.00 | 65.67 | 69.67 | 67.67 |
| rot5 | 46.67 | 66.67 | 68.33 | 70.00 |
| rot10 | 33.00 | 65.67 | 68.67 | 68.33 |
| shift8 | 33.67 | 64.33 | 65.33 | 71.67 |
| **均值** | 34.57 | 63.91 | **66.52** | **67.57** |

三条结论：

1. **可微几何模块能替代 SIFT+RANSAC，代价 ≈ −1.1pp**（local 均值 66.52 vs 经典 67.57），且 **7 档里 4 档反超**；
   明显缺口只在 shift8（−6.3pp）与 clean（−3.7pp）。
2. **重排器本体判别力与经典持平**：候选择对的条件 Top-1 我们 63~70%、经典 62~72%；
   几何分在候选集内高度可判（正 ~0.79 / 负 ~0.05，逐查询成对 AUC 99.6%）。
3. **残差几乎全在初筛召回**：`crop0.7off` 的 recall@300 仅 91.33%，是唯一大缺口；融合初筛分基本无增益（最优 w 多为 1.0 纯几何）。

**定位**：可微几何重排器是**"持平经典的等价替代"（工程贡献）**，不是"反超经典"的方法贡献 —— σ/temp 网格扫描（§6.4）已证明缺口不是参数没调好。这也正是**路线 B** 的定位（见开头）。

### 6.4 参数敏感性：几何模块 σ / temp 网格扫描（探索 1，**σ 假设证伪**）

脚本 [\_scan_geo_params.py](file:///d:/py/AURA/aura/runs/_scan_geo_params.py)（纯推理、不重训；temp 外层算一次、σ 内层只重算 `geom_score`），口径与 §6.3 逐项对齐（`n_query=300` 同子集、`features=local`、初筛 fuse、grid 32²）。结果 [scan_geo_params_local.json](file:///d:/py/AURA/aura/runs/scan_geo_params_local.json)。

| temp \ sigma | 0.02 | 0.04 | 0.06 | **0.08** |
| --- | --- | --- | --- | --- |
| 0.01 | 61.05 / 61.67 | 64.57 / 64.76 | 65.76 / 65.90 | **66.14 / 66.38** |
| 0.03 | 62.71 / 63.05 | 64.95 / 65.38 | 66.05 / 66.33 | **66.52 / 66.90** |
| 0.05 | 62.90 / 63.14 | 65.38 / 65.62 | 66.10 / 66.52 | **66.52 / 66.76** |
| 0.1 | 58.57 / 59.14 | 63.52 / 63.62 | 64.81 / 65.00 | **65.19 / 65.24** |

（每格 = `纯几何 T1 均值 / best_w 融合 T1 均值`，7 档平均。）

三条结论：

1. **σ 假设证伪**：σ **越大越好**，0.08 在四个 temp 下都是各自最优；"σ=0.08 过大、软内点失效"不成立。
2. **temp 影响很小**：0.01 / 0.03 / 0.05 近乎无差（最优均值同为 66.52），0.1 明显掉。
3. **调参补不上缺口**：全网格最优 = 纯几何 **66.52** / 融合 **66.90**，仍 < 经典 **67.57** ⇒ §6.3 的 −1.1pp 不是参数没调好；shift8 仍是唯一系统性缺口（65.33 vs 经典 71.67），其余 6 档已全部 ≥ 经典。

### 6.5 open-set 拒识协议（HashVAE 与两家免训练基线都没有的指标）

闭集协议隐含"查询源图一定在库里"；真实取证里多数查询源头**不在库中**，须先判"库里没有归宿"。
实现 [eval.py `evaluate_openset`](file:///d:/py/AURA/aura/eval.py)（**追加，不改现有函数**），脚本 [\_eval_openset.py](file:///d:/py/AURA/aura/runs/_eval_openset.py)，读数 `aura/runs/openset_eval.json`。
**退化档**：`evaluate_openset_levels`（同样追加），脚本加 `--levels all`，读数 `aura/runs/openset_deg.json`（12 档）。

口径：test 200 条真实视频按 `identity` 切**闭集 150 / 开集 50**；图库 = 闭集 1200 张干净真帧；伪造查询全局确定性子采样 2000 张 → 闭集 1496 / 开集 504；
拒识打分 = 查询对全库的 top-1 余弦（闭集标 1 / 开集标 0），阈值在开集分布的目标 FPR 分位上标定。

| 变体 | 闭集帧 T1 | 闭集帧 T5 | 闭集视频 T1 | 拒识 AUC | TPR@FPR=1% | 相似度 闭集 / 开集 |
| --- | --- | --- | --- | --- | --- | --- |
| **I-best**（2048 维，训练） | 60.36% | 89.10% | 99.73% | **99.75%** | 98.73% | 0.9891 / 0.7172 |
| **r8_norm**（192 维，零训练） | 61.03% | 88.90% | 99.73% | **99.73%** | 98.86% | 0.9842 / 0.7207 |

三条结论：

1. 协议可跑通、闭集块与标准协议同量级（图库缩到 1200，Top-1 略高于全量协议的 59.68%，属预期）；开集块 Top-K **恒为 0.00%**。
2. **训练模型在拒识上仍没超过零训练 192 维 r8**（99.75 vs 99.73）—— 与帧级硬线结论一致。
3. **干净档下开集拒识近乎平凡**：开集查询的源视频整条缺席，其 top-1 相似度（~0.72）天然远低于闭集（~0.99），AUC ≈ 99.7% 是**易任务**、两法**无区分度**。

**退化档重测**（`--levels all`，图库仍干净、逐档只退化查询）：

| 档位 | I-best AUC / TPR@1% | r8 AUC / TPR@1% | 档位 | I-best AUC / TPR@1% | r8 AUC / TPR@1% |
| --- | --- | --- | --- | --- | --- |
| clean | 99.75 / 98.73 | 99.73 / 98.86 | crop0.8 | 87.41 / 55.88 | 82.02 / 18.85 |
| jpeg50 | 99.75 / 98.80 | 99.73 / 98.86 | crop0.7off | 67.04 / 7.55 | 61.53 / 14.10 |
| jpeg20 | 99.74 / 98.53 | 99.73 / 98.86 | zoom1.15 | 98.22 / 81.28 | 97.53 / 76.74 |
| scale0.5 | 99.75 / 98.66 | 99.73 / 98.86 | rot5 | 99.55 / 94.18 | 99.66 / 98.40 |
| crop0.5 | **55.84 / 6.15** | **51.22 / 1.94** | rot10 | 96.87 / 64.64 | 97.71 / 76.00 |
| reencode | 99.75 / 98.60 | 99.73 / 98.86 | shift8 | 99.58 / 94.52 | 99.58 / 96.59 |

**结论**：① **光度六档下拒识几乎不变**（AUC 恒 99.7%），与"判别信号在低频"一致；② **几何档才是真正的评测点** —— `crop0.5` 使 AUC 崩到 **55.84**（TPR 仅 6.15）；③ 退化档下训练模型与零训练 r8 **各有胜负、量级相当**，**训练在拒识维度同样没有稳定增益**。

