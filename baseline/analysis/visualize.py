"""可视化：哈希码流形(UMAP)、损失曲线、重构样例。

用法（在项目根目录 d:\py\AURA 下运行）：
    python -m baseline.analysis.visualize --config configs/baseline.yaml
"""
import argparse
import json
import os

# 必须在 import torch 之前设置：4GB 显存下开启可扩展显存段，缓解碎片导致的 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")            # 无显示环境（服务器/命令行）下也能出图
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.data import build_datasets
from baseline.models import HashVAE
from baseline.utils import ensure_dir, get_device, load_config, resolve_path, set_seed


@torch.no_grad()
def collect_hashes(model, loader, device):
    """收集整个数据集的硬哈希码与标签。"""
    model.eval()
    hashes, labels = [], []
    for batch in loader:
        img = batch["image"].to(device)
        mask = batch.get("mask")
        out = model(img, mask=None if mask is None else mask.to(device),
                    need_recon=False, need_pred=False)   # 只要哈希码
        hashes.append(out["code"].cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(hashes, 0), np.concatenate(labels, 0)


def plot_umap(hashes, labels, out_path):
    """用 UMAP 把 128 维哈希码降到 2 维，观察真假/身份是否可分。"""
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(hashes)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="coolwarm", s=6, alpha=0.6)
    plt.colorbar(sc, label="label (0=real, 1=fake)")
    plt.title("Hash Code Manifold (UMAP)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"已保存: {out_path}")


def plot_loss_curves(history_path, out_path):
    """把训练 history.json 里的 loss 画成曲线。"""
    if not os.path.exists(history_path):
        print(f"[warn] 未找到 {history_path}，跳过 loss 曲线")
        return
    with open(history_path, "r", encoding="utf-8") as f:
        hist = json.load(f)
    s1, s2 = hist.get("stage1", []), hist.get("stage2", [])
    plt.figure(figsize=(10, 4))
    if s1:
        plt.subplot(1, 2, 1)
        plt.plot([d["recon"] for d in s1], label="recon")
        plt.plot([d["entropy"] for d in s1], label="entropy")
        plt.title("Stage1"); plt.xlabel("epoch"); plt.legend()
    if s2:
        plt.subplot(1, 2, 2)
        for k in ["recon", "quad", "cls"]:
            if k in s2[0]:
                plt.plot([d[k] for d in s2], label=k)
        plt.title("Stage2"); plt.xlabel("epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="AURA 可视化")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg)
    name = cfg.get("name", "baseline")
    res_dir = ensure_dir("results/")

    # loss 曲线（来自训练 history）
    plot_loss_curves(resolve_path(os.path.join(cfg.get("save_dir", "checkpoints/"), f"{name}_history.json")),
                     os.path.join(res_dir, f"{name}_loss_curve.png"))

    # 哈希流形（需要模型权重）
    ckpt = resolve_path(args.ckpt or os.path.join(cfg.get("save_dir", "checkpoints/"), f"{name}.pth"))
    if not os.path.exists(ckpt):
        print(f"[warn] 未找到权重 {ckpt}，跳过 UMAP")
        return
    datasets, _ = build_datasets(cfg)
    test_ds = datasets["test"]
    test_ds.warm_mask_cache()                       # 建好掩膜缓存并释放解析网络，避免争显存
    loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=int(cfg.get("num_workers", 0)))
    model = HashVAE.build_from_config(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    hashes, labels = collect_hashes(model, loader, device)
    plot_umap(hashes, labels, os.path.join(res_dir, f"{name}_hash_manifold.png"))


if __name__ == "__main__":
    main()
