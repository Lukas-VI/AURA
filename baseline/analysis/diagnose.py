"""诊断实验：验证“人工裁剪依赖 / 背景唯一性 / 频域敏感度”。

4 个实验：
    1) crop_dependency        —— 整体裁剪平移对检索性能的影响
    2) background_sensitivity —— 背景被攻击后哈希码的变化量
    3) face_sensitivity       —— 人脸被换脸后哈希码的变化量（应与背景攻击对比）
    4) frequency_sensitivity  —— 背景不同频段被滤除后哈希码的变化量

用法（在项目根目录 d:\py\AURA 下运行）：
    python -m baseline.analysis.diagnose --config configs/baseline.yaml
"""
import argparse
import json
import os

# 必须在 import torch 之前设置：4GB 显存下开启可扩展显存段，缓解碎片导致的 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from baseline.data import FaceBackgroundSplit, SyntheticFaceSwap, build_datasets
from baseline.eval import evaluate_retrieval
from baseline.models import HashVAE
from baseline.utils import ensure_dir, get_device, load_config, resolve_path, set_seed


def shift_image(img_bgr, dx, dy):
    """整幅图平移 (dx,dy) 像素，用于模拟裁剪偏移。"""
    h, w = img_bgr.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])           # 平移矩阵
    return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def band_filter(img_bgr, mask, band):
    """只对背景做频域带通/带阻，前景(人脸)保持原样。band in {'low','mid','high'}。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))             # 频域
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)       # 到中心的频率半径
    rmax = min(cy, cx)
    if band == "low":                                   # 只保留低频
        keep = r <= rmax * 0.25
    elif band == "high":                                # 只保留高频
        keep = r >= rmax * 0.5
    else:                                               # 保留中频
        keep = (r > rmax * 0.15) & (r < rmax * 0.6)
    f = f * keep
    filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(f)))   # 逆变换回空域
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    out = img_bgr.copy()
    for c in range(3):
        out[..., c] = (img_bgr[..., c] * mask + filtered * (1 - mask)).astype(np.uint8)
    return out


@torch.no_grad()
def _encode(model, imgs, device, bs=4):
    """分块编码，避免一次性把整批图送进 GPU 造成显存溢出（4GB 卡尤其重要）。"""
    outs = []
    for s in range(0, len(imgs), bs):
        batch = torch.stack(imgs[s:s + bs]).to(device)
        out = model(batch, need_recon=False, need_pred=False)   # 只要哈希码
        outs.append(out["code"].cpu())
    return torch.cat(outs, dim=0)


class _TransformLoader:
    """把一个 Subset 包成“对每张图先做 img_fn 变换再张量化”的迭代器。"""

    def __init__(self, dataset, indices, img_fn, batch_size=4):
        self.dataset, self.indices, self.img_fn, self.bs = dataset, list(indices), img_fn, batch_size

    def __len__(self):
        return (len(self.indices) + self.bs - 1) // self.bs

    def __iter__(self):
        for s in range(0, len(self.indices), self.bs):
            idxs = self.indices[s:s + self.bs]
            images, ids, source_ids, kinds = [], [], [], []
            for i in idxs:
                smp = self.dataset.samples[i]
                bgr = self.dataset._read(smp["path"])        # 真/假样本都直接读自己的图
                bgr = self.img_fn(bgr)                       # 施加诊断变换
                images.append(self.dataset.to_tensor(bgr))   # 统一张量化
                ids.append(smp["id"]); source_ids.append(smp["source_id"]); kinds.append(smp["kind"])
            yield {"image": torch.stack(images), "id": ids, "source_id": source_ids, "kind": kinds}


def exp_crop_dependency(model, dataset, device, offsets=(0, 5, 10, 20)):
    """实验 1：把测试集整体平移若干像素，观察 Top-5 检索准确率的变化。"""
    real = [i for i, s in enumerate(dataset.samples) if s["kind"] == "real"]
    fake = [i for i, s in enumerate(dataset.samples) if s["kind"] == "fake"]
    results = {}
    for off in offsets:
        fn = (lambda b, o=off: shift_image(b, o, o))     # 固定偏移闭包
        db = _TransformLoader(dataset, real, fn)
        qy = _TransformLoader(dataset, fake, fn)
        ret = evaluate_retrieval(model, db, qy, device, top_k=(5,))
        results[f"offset_{off}"] = ret["top5"]
        print(f"[crop] offset={off:>2} -> Top-5 = {ret['top5']:.4f}")
    return results


def exp_background_sensitivity(model, dataset, device, modes=("blur", "color", "jpeg")):
    """实验 2：对真实图背景做攻击，统计哈希码的平均汉明距离（越大越敏感）。"""
    splitter, faker = dataset.splitter, dataset.faker
    reals = [i for i, s in enumerate(dataset.samples) if s["kind"] == "real"][:16]
    orig = [dataset.to_tensor(dataset._read(dataset.samples[i]["path"])) for i in reals]
    h0 = _encode(model, orig, device)
    results = {}
    for mode in modes:
        attacked = []
        for i in reals:
            bgr = dataset._read(dataset.samples[i]["path"])
            attacked.append(dataset.to_tensor(faker.background_attack(bgr, mode=mode)))
        h1 = _encode(model, attacked, device)
        dist = torch.nn.functional.pairwise_distance(h0, h1).mean().item()
        results[mode] = dist
        print(f"[bg-sens] {mode:>5} -> mean distance = {dist:.4f}")
    return results


def exp_face_sensitivity(model, dataset, device):
    """实验 3：对真实图前景换脸，统计哈希码平均距离（AURA 希望它小于背景攻击）。"""
    faker = dataset.faker
    reals = [i for i, s in enumerate(dataset.samples) if s["kind"] == "real"][:16]
    orig, swapped = [], []
    n = len(reals)
    for k, i in enumerate(reals):
        bgr = dataset._read(dataset.samples[i]["path"])
        donor = dataset._read(dataset.samples[reals[(k + 7) % n]]["path"])  # 同身份供体，仅改人脸
        orig.append(dataset.to_tensor(bgr))
        swapped.append(dataset.to_tensor(faker.swap(bgr, donor)))
    h0 = _encode(model, orig, device)
    h1 = _encode(model, swapped, device)
    dist = torch.nn.functional.pairwise_distance(h0, h1).mean().item()
    print(f"[face-sens] faceswap -> mean distance = {dist:.4f}")
    return {"faceswap": dist}


def exp_frequency_sensitivity(model, dataset, device, bands=("low", "mid", "high")):
    """实验 4：只对背景做频段滤波，统计哈希码平均距离。"""
    splitter = dataset.splitter
    reals = [i for i, s in enumerate(dataset.samples) if s["kind"] == "real"][:16]
    orig = [dataset.to_tensor(dataset._read(dataset.samples[i]["path"])) for i in reals]
    h0 = _encode(model, orig, device)
    results = {}
    for band in bands:
        filt = []
        for i in reals:
            bgr = dataset._read(dataset.samples[i]["path"])
            m = splitter.mask(bgr)
            filt.append(dataset.to_tensor(band_filter(bgr, m, band)))
        h1 = _encode(model, filt, device)
        dist = torch.nn.functional.pairwise_distance(h0, h1).mean().item()
        results[band] = dist
        print(f"[freq] band={band:>4} -> mean distance = {dist:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="AURA 基线诊断实验")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg)

    datasets, _ = build_datasets(cfg)
    test_ds = datasets["test"]

    model = HashVAE.build_from_config(cfg).to(device)
    ckpt = resolve_path(args.ckpt or os.path.join(cfg.get("save_dir", "checkpoints/"),
                                                  f"{cfg.get('name', 'baseline')}.pth"))
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"已加载权重: {ckpt}")

    results = {}
    # 顺序执行并在每个实验后清空显存缓存，避免 4GB 卡上碎片累积导致 OOM
    results["crop_dependency"] = exp_crop_dependency(model, test_ds, device)
    torch.cuda.empty_cache()
    results["background_sensitivity"] = exp_background_sensitivity(model, test_ds, device)
    torch.cuda.empty_cache()
    results["face_sensitivity"] = exp_face_sensitivity(model, test_ds, device)
    torch.cuda.empty_cache()
    results["frequency_sensitivity"] = exp_frequency_sensitivity(model, test_ds, device)
    out_path = os.path.join(ensure_dir("results/"), f"{cfg.get('name', 'baseline')}_diagnose.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"诊断结果已保存: {out_path}")


if __name__ == "__main__":
    main()
