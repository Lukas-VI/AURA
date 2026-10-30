"""方案 B 第 2 步：用【已训好的 E 轮权重、不重训】跑几何扰动协议，与零训练侧对照。

背景：零训练 32x32 像素匹配器在几何扰动下大幅崩盘（crop0.8 10.5% / zoom1.15 25.7% /
rot10 25.9% / shift8 30.8% / rot5 41.0%，见 probe_protocol_ceiling.json）。现在要看
同一个扰动下深度模型是"也崩"还是"守住"：
    守住  -> 训练增量价值直接证实，无需重训；
    也崩  -> 说明该模型训练时没见过几何增强，需带几何增强重训一轮（方案 B 第 3 步）。
注意：本轮模型训练时的增强只有水平翻转 / 降采样 / JPEG / 模糊 / 噪声 / 亮度，**没有旋转、
平移、裁剪**，所以"也崩"是预期内的公平性缺口，不是能力结论。

协议与既有一致：图库干净且只编码一次，只对查询做确定性几何扰动。

用法（项目根目录 d:\py\AURA 下）：
    $env:CUDA_DEVICE_ORDER="PCI_BUS_ID"; $env:CUDA_VISIBLE_DEVICES="1";
    $env:PYTHONPATH="d:\py\AURA";
    E:\conda\envs\pytorch\python.exe -u aura\runs\_eval_geom_model.py
"""
import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from aura.data import GEOM_LEVELS, ContrastiveDataset
from aura.encoder import SourceEncoder
from aura.eval import evaluate_degradation, format_degradation
from baseline.data.dataset import build_ffpp_index
from baseline.utils import get_device, load_config

DEFAULT_CFG = os.path.abspath("aura/configs/aura_full.yaml")
DEFAULT_CKPT = "D:/py/AURA/aura/runs/aura_full_fine_e6.pth"      # E 轮（source_id 细粒度）e6
EVAL_BS = 8                                                      # 8 够快，前向无梯度检查点，峰值仍安全


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default="D:/py/AURA/aura/runs/geom_model_e6.json")
    ap.add_argument("--bs", type=int, default=EVAL_BS)
    ap.add_argument("--n_query", type=int, default=2000,
                    help="查询子采样数（确定性抽样，可复现）。默认 2000 约 20min；"
                         "0 = 全量 8000（约 77min），仅在需要最终判定/与历史读数严格可比时使用。")
    args = ap.parse_args()

    cfg = load_config(os.path.abspath(args.config))
    cfg["eval_batch_size"] = args.bs
    dev = get_device(cfg)
    print(f"device={dev} | config={args.config} | ckpt={args.ckpt}", flush=True)

    model = SourceEncoder.build_from_config(cfg).to(dev)
    # strict=False：K 轮权重带局部描述子头（local_head.*），旧轮次权重没有 —— 两边都能载。
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location=dev), strict=False)
    model.eval()
    if missing or unexpected:
        print(f"[load] strict=False | 缺键 {len(missing)} / 多余键 {len(unexpected)}", flush=True)
    print(f"参数 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)

    index = build_ffpp_index(cfg)
    real = [s for s in index["test"] if s["kind"] == "real"]
    fake = [s for s in index["test"] if s["kind"] == "fake"]
    if args.n_query and args.n_query < len(fake):
        fake = random.Random(0).sample(fake, args.n_query)   # 固定种子 -> 可复现，且各档位查询集一致
    size = int(cfg["image_size"])
    db_loader = DataLoader(ContrastiveDataset(real, image_size=size, train=False, seed=0),
                           batch_size=args.bs, shuffle=False, num_workers=0)
    print(f"图库 {len(real)} / 查询 {len(fake)}", flush=True)

    levels = ("clean",) + tuple(GEOM_LEVELS)
    t0 = time.time()
    res = evaluate_degradation(model, db_loader, fake, cfg, dev, top_k=(1, 5), levels=levels)
    print(format_degradation(res), flush=True)
    print(f"[time] {(time.time() - t0) / 60:.1f} min", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
