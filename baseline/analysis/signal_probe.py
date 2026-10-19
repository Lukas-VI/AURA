"""零训练判别性探针：冻结编码器，只测【前景哈希特征 h_f】里到底有没有真假信息。

背景（为何要做）：三代实现下检测损失 cls 恒为 2·ln2 = ln4 ≈ 1.3863（loss_c 是两个 BCE 项
之和，DDB 输出恒 ~0.5 时每项为 ln2）。有两种成因，
修法完全相反：
    (a) 码里根本没有真假信息（四元组前景项把真/假前景压到相近码位、C23 重压缩抹掉高频
        伪影、容量太小）—— 任何损失编排都救不了，只能提容量 / 提分辨率；
    (b) 码里有信息，但 stage2 的联合训练动力学让 DDB 学不到 —— 加 detach / 调损失调度即可。
判别方法：把训练好的编码器冻结，抽测试集每张图的前景哈希特征 h_f，离线训一个简单分类头，
看测试 AUC。显著 > 0.5 → (b)；仍 ≈ 0.5 → (a)。

为对齐，本脚本同时抽背景哈希特征 h_b 作为对照（预期 h_b 区分真假的能力弱于 h_f）。

用法（在项目根目录 d:\py\AURA 下运行）：
    python -m baseline.analysis.signal_probe --config configs/baseline_parse.yaml
"""
import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from baseline.data import build_datasets
from baseline.models import HashVAE
from baseline.utils import ensure_dir, get_device, load_config, resolve_path, set_seed


@torch.no_grad()
def extract_features(model, dataset, device, batch_size, num_workers=0, verbose=True):
    """遍历整个数据集，返回 (h_f, h_b, label, group)。group 用视频号，避免帧级泄漏。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    feats_f, feats_b, labels, groups = [], [], [], []
    for i, batch in enumerate(loader):
        img = batch["image"].to(device)
        mask = batch.get("mask")
        out = model.encode(img, mask=None if mask is None else mask.to(device))
        feats_f.append(out["h_f"].cpu().numpy())
        feats_b.append(out["h_b"].cpu().numpy())
        labels.append(batch["label"].numpy())
        groups.extend(batch["identity"])
        if verbose and (i + 1) % 20 == 0:
            print(f"  [probe] {(i + 1) * loader.batch_size}/{len(dataset)}")
    return (np.concatenate(feats_f), np.concatenate(feats_b),
            np.concatenate(labels), np.asarray(groups))


def probe(name, X, y, groups, seed=42):
    """按视频号分组切 train/test，训练两个简单头并报 AUC。"""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    ytr, yte = y[tr], y[te]

    res = {"feature": name, "dim": int(X.shape[1]),
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "test_pos_rate": float(yte.mean())}

    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xtr, ytr)
    res["linear_train_auc"] = float(roc_auc_score(ytr, lr.predict_proba(Xtr)[:, 1]))
    res["linear_test_auc"] = float(roc_auc_score(yte, lr.predict_proba(Xte)[:, 1]))

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
    mlp.fit(Xtr, ytr)
    res["mlp_train_auc"] = float(roc_auc_score(ytr, mlp.predict_proba(Xtr)[:, 1]))
    res["mlp_test_auc"] = float(roc_auc_score(yte, mlp.predict_proba(Xte)[:, 1]))
    return res


def main():
    parser = argparse.ArgumentParser(description="AURA 前景/背景哈希特征的真假信息探针（零训练）")
    parser.add_argument("--config", default="configs/baseline_parse.yaml")
    parser.add_argument("--ckpt", default=None, help="权重路径，默认 checkpoints/<name>.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help=">0 时只抽前 N 个样本（冒烟用）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg)

    datasets, _ = build_datasets(cfg)
    test_ds = datasets["test"]
    test_ds.warm_mask_cache()
    if args.limit > 0:
        test_ds.samples = test_ds.samples[:args.limit]
    print(f"探针样本数: {len(test_ds)}（真实+伪造混合）")

    model = HashVAE.build_from_config(cfg).to(device)
    ckpt = resolve_path(args.ckpt or os.path.join(cfg.get("save_dir", "checkpoints/"),
                                                 f"{cfg.get('name', 'baseline')}.pth"))
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"已加载权重: {ckpt}")

    h_f, h_b, y, groups = extract_features(model, test_ds, device, args.batch_size,
                                           num_workers=int(cfg.get("num_workers", 0)))
    print(f"特征形状: h_f={h_f.shape}, h_b={h_b.shape}, y 正例率={y.mean():.3f}")

    results = [probe("h_f (前景哈希特征，DDB 的输入)", h_f, y, groups),
               probe("h_b (背景哈希特征，对照)", h_b, y, groups)]
    print("\n=== 探针结果（按视频号分组，帧级无泄漏）===")
    for r in results:
        print(f"{r['feature']}: dim={r['dim']} | 线性 train/test AUC = "
              f"{r['linear_train_auc']:.4f}/{r['linear_test_auc']:.4f} | "
              f"MLP train/test AUC = {r['mlp_train_auc']:.4f}/{r['mlp_test_auc']:.4f}")

    # 裁决只看 h_f —— DDB 读的就是它；h_b 仅作对照
    auc_hf = results[0]["linear_test_auc"]
    verdict = ("(b) h_f 码里有真假信息 -> 属训练动力学/编排问题，优先加 detach / 调损失调度"
               if auc_hf > 0.55 else
               "(a) h_f 码里没有真假信息 -> 属信息缺失，只能提容量 / 提分辨率")
    print(f"\n裁决：h_f 线性测试 AUC={auc_hf:.4f}（h_b 对照={results[1]['linear_test_auc']:.4f}）-> {verdict}")

    out_dir = ensure_dir("results/")
    out_path = os.path.join(out_dir, f"{cfg.get('name', 'baseline')}_signal_probe.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": ckpt, "n_samples": int(len(test_ds)),
                   "results": results, "verdict": verdict}, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {out_path}")


cfg_num_workers = 0
if __name__ == "__main__":
    main()
