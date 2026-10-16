"""检测评估：在真实/伪造混测试集上计算 AUC 与准确率。"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


@torch.no_grad()
def evaluate_detection(model, loader, device, threshold: float = 0.5):
    """返回 (AUC, ACC)。loader 需同时包含真实(label=0)与伪造(label=1)样本。

    检测概率来自检测块 DDB，而 DDB 只读取【前景哈希特征】（论文 3.4.1）。
    """
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        img = batch["image"].to(device)
        mask = batch.get("mask")                     # 逐样本前景掩膜（诊断用的临时 loader 可能没有）
        pred = model(img, mask=None if mask is None else mask.to(device),
                     need_recon=False)["pred"]       # 只需检测分支，省显存
        all_preds.append(pred.cpu().view(-1))
        all_labels.append(batch["label"].view(-1))

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)

    auc = roc_auc_score(labels, preds)                        # AUC：对阈值不敏感
    acc = accuracy_score(labels, (preds > threshold).astype(int))  # ACC：阈值 0.5
    return float(auc), float(acc)
