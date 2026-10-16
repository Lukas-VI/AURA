"""baseline 包：HashVAE 简化版基线的自包含实现。

基线论文并未开源，这里按论文《Deep hashing framework for proactively deepfake
retrieval and detection》(Liu et al., 2026) 的描述重写一版简化实现，用于建立实验对照。

包内所有相对路径（配置、权重、日志）都以本包目录为基准解析，因此可以用
``python -m baseline.train_baseline`` 在项目根目录下运行，产物落在 baseline/ 内。
"""
