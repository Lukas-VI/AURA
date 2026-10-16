"""AURA 基线测试脚本：加载训练好的模型，评估检索与检测指标。

用法（在项目根目录 d:\py\AURA 下运行）：
    python -m baseline.test_baseline --config configs/baseline.yaml
"""
import argparse
import json
import os

# 必须在 import torch 之前设置：4GB 显存下开启可扩展显存段，缓解碎片导致的 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader, Subset

from baseline.data import build_datasets
from baseline.eval import evaluate_detection, evaluate_retrieval
from baseline.models import HashVAE
from baseline.utils import ensure_dir, get_device, load_config, resolve_path, set_seed


def _indices_by_kind(dataset, kind):
    """返回数据集中特定类型（real/fake）样本的下标。"""
    return [i for i, s in enumerate(dataset.samples) if s["kind"] == kind]


def main():
    parser = argparse.ArgumentParser(description="AURA 基线测试")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--ckpt", default=None, help="权重路径，默认取 checkpoints/<name>.pth")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg)

    # ---------- 数据（切分逻辑与训练一致，seed/规模相同则完全复现，无泄漏） ----------
    datasets, _ = build_datasets(cfg)
    test_ds = datasets["test"]
    # 先建好测试集的解析掩膜缓存再释放解析网络，避免它与 HashVAE 争 4GB 显存
    test_ds.warm_mask_cache()

    real_idx = _indices_by_kind(test_ds, "real")    # 数据库：真实图
    fake_idx = _indices_by_kind(test_ds, "fake")    # 查询：伪造图
    print(f"测试集：真实 {len(real_idx)} 张（数据库），伪造 {len(fake_idx)} 张（查询）")

    nw = int(cfg.get("num_workers", 0))
    db_loader = DataLoader(Subset(test_ds, real_idx), batch_size=cfg["batch_size"],
                           shuffle=False, num_workers=nw)
    query_loader = DataLoader(Subset(test_ds, fake_idx), batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=nw)
    mixed_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=nw)

    # ---------- 模型 ----------
    model = HashVAE.build_from_config(cfg).to(device)
    ckpt = resolve_path(args.ckpt or os.path.join(cfg.get("save_dir", "checkpoints/"),
                                                  f"{cfg.get('name', 'baseline')}.pth"))
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"已加载权重: {ckpt}")

    # ---------- 检索评估 ----------
    # 主指标：用【背景哈希码】检索（论文 Algorithm 1 情形1的第一步，对应 AURA 核心假设）
    ret = evaluate_retrieval(model, db_loader, query_loader, device, top_k=(1, 5), key="bg")
    print(f"[背景码检索] Top-1: {ret['top1']:.4f}, Top-5: {ret['top5']:.4f}")
    # 对照：用前后景拼接的完整哈希码检索
    ret_full = evaluate_retrieval(model, db_loader, query_loader, device, top_k=(1, 5), key="full")
    print(f"[完整码检索] Top-1: {ret_full['top1']:.4f}, Top-5: {ret_full['top5']:.4f}")

    # ---------- 检测评估 ----------
    auc, acc = evaluate_detection(model, mixed_loader, device)
    print(f"AUC: {auc:.4f}, ACC: {acc:.4f}")

    # ---------- 保存结果 ----------
    res_dir = ensure_dir("results/")
    out = {"checkpoint": ckpt,
           "retrieval_bg_top1": ret["top1"], "retrieval_bg_top5": ret["top5"],
           "retrieval_full_top1": ret_full["top1"], "retrieval_full_top5": ret_full["top5"],
           "AUC": auc, "ACC": acc, "num_query": ret["n"]}
    out_path = os.path.join(res_dir, f"{cfg.get('name', 'baseline')}_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"指标已保存: {out_path}")


if __name__ == "__main__":
    main()
