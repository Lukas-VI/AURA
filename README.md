# AURA — 深度伪造主动检索（基线阶段）

**AURA** (Attack-Unsupervised Retrieval of Authentic backgrounds)：给定一张深度伪造图，
检索出它对应的**真实源图**（利用背景作为来源指纹），同时输出真假判别。

当前处于第一阶段：复现论文 *Deep hashing framework for proactively deepfake retrieval and
detection* (Liu et al., 2026) 的 HashVAE。由于该基线**并未开源**，本仓库按论文描述
**重写了一版简化实现**，放在 `baseline/` 下，作为后续自建模型的实验对照。

## 1. 目录结构

```
AURA/
├── README.md
├── requirements.txt
├── aura/                     # 【占位】后期 AURA 自建模型放这里（目前只有 __init__.py）
└── baseline/                 # 简化版 HashVAE 基线（自包含包）
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
```

> `baseline/` 是自包含包：配置路径、权重、日志都以**包目录**为基准解析，
> 因此统一用 `python -m baseline.xxx` 在**项目根目录**下运行即可。

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
| 掩膜由人脸检测 + 注意力擦除 top-n 生成 | **BiSeNet 19 类解析**：前景 = 类 1–18（一切人语义：脸/皮肤/头发/衣服/脖子），背景 = 类 0（纯场景）；掩膜随样本从 dataset 流入模型 | 默认 `fg_mask_source: 'parse'`；几何椭圆保留为 `'geometry'` 消融/兜底。论文用检测+擦除造掩膜，本实现改用现成 SimSwap 解析权重（零额外依赖），实测比椭圆显著提升检索（见 5.1） |
| 式7：哈希特征经 softmax | 逐比特映射到 (0,1)（BN+Tanh） | 按字面在码维做 softmax 会挤成 one-hot → 哈希坍缩，故用逐比特映射 |
| M=5 分支 / D=5 RB / 64 通道；hash 2048 位 | 2 分支 / 2 RB / base 32 / aff 256；hash 128 位 | 早期受 4GB 显存（RTX 2050）限制的简化档；现已换到 22GB RTX 2080 Ti，容量放大列为下一步 |
| 数据集 FF++ / Celeb-DF / DFDC | FF++ C23 抽帧（D:/ffpp_frames） | 已换成真实伪造帧 |
| 标签 real=1 / fake=0 | real=0 / fake=1 | 仅符号约定差异 |
| 两阶段检索 Algorithm 1 | 检索评估用背景码 `C_b` 汉明距离 | 对应论文情形(1) 第一步；与 AURA“背景=来源指纹”假设一致 |
| faceswap/simswap 生成伪造图 | FF++ 真实伪造帧（q+2）；`q+1` 仍用自包含合成换脸 | q+1 只用于前景项，合成即可 |

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

**检测（仍未解锁）**：`cls` 三代实现（单哈希 / 几何椭圆 / BiSeNet 解析）**恒为 ln2≈1.3863**
（DDB 输出恒 ~0.5），本轮检测 AUC = **0.5166 ≈ 随机**。诊断结论：

- 检测块 DDB 的**结构正确**——固定前景哈希、单独训 DDB，400 步内即可把损失从 0.75 降到 0.001；
- 绕开四元组/最小熵做**纯监督 BCE**，训练/测试 AUC 仍≈随机（train 0.482 / test 0.467）；
- 只有对**固定的一对**图训练时损失才下降（能记忆、不能泛化）。

归因（**已修正，旧结论作废**）：早期把原因归为"合成换脸是干净贴脸、前景无伪影"，
并建议换真实 Deepfake 数据集 —— **换完 FF++ 真实伪造帧后 `cls` 依旧恒为 ln2，该归因被推翻**。
真正的原因更可能是**目标冲突**：DDB 只读**前景**哈希，而前景要同时满足四元组前景项
（把"真 A 的前景"与"假 A 的前景"拉近）与最小熵/比特平衡（压到均匀分布），于是真假前景
被压到相近码位，DDB 拿不到可分信号。掩膜改的是**背景纯度**（作用于检索），对前景检测目标
无能为力。受"不改 DDB 结构（论文权威）"的约束，该冲突需从**损失编排/训练范式**侧解。

**下一步**：① 用 22GB 2080 Ti 兑现容量红利（base 32→64、aff 256→512、stage2 batch→16）重跑；
② 试"前景哈希 stop-gradient 后再喂 DDB"以验证冲突假说；③ 解释"完整码仍优于背景码"。
详见 [docs/AURA\_研究方向评估与实现方案.md](file:///d:/py/AURA/docs/AURA_研究方向评估与实现方案.md) §11。

