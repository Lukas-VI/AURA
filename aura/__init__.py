"""aura 包：AURA 自建模型（单编码器 + 全图对比学习）。

组成：
    encoder.py  SourceEncoder —— 整图输入的单编码器（全量规格主干）
    losses.py   SupCon（带跨批内存 XBM） + VICReg 式方差铰链
    data.py     退化增强 + 同源分组批采样
    eval.py     检索评估（余弦 / 符号二值化汉明，协议同 baseline）
    train.py    单阶段对比学习训练入口
"""
