"""从 FaceForensics++ (C23) 原始视频中抽帧，并按人脸裁剪，产出 AURA 可用的图像数据集。

背景
----
`D:\\ffpp\\FaceForensics++_C23\\` 下是 7 个视频文件夹 + csv：
    original(1000 真实) / Deepfakes / Face2Face / FaceSwap / NeuralTextures /
    FaceShifter / DeepFakeDetection(各 1000 伪造)
视频名形如 `01_02__meeting_serious__YVGY8LOK.mp4`，其中 `01_02` 是【身份_片段】；
同一身份片段在 original 与各伪造文件夹里【同名】，因此可以一一配对成
"伪造图 -> 真实源图" 的检索样本。

本脚本做三件事
--------------
1. 依次读取每个视频，均匀抽取 N 帧（默认 8 帧）；
2. 用 OpenCV 自带的 YuNet 人脸检测器（cv2.FaceDetectorYN）检出最大人脸，
   以人脸为中心外扩 margin 倍裁成正方形，缩放到 --size（默认 256）后存为 jpg；
   这样"人脸大致居中"，与本项目"前后景用居中椭圆分离"的先验一致；
3. 输出目录结构 + manifest.csv（记录 folder/stem/frame/video/face_box/label），
   便于后续把该数据集接进 AURADataset。

输出结构
--------
    <out>/
      original/<stem>/f000.jpg …
      Deepfakes/<stem>/f000.jpg …
      …
      manifest.csv

用法（项目根目录 d:\py\AURA 下运行）
-----------------------------------
    # 先小规模试跑（每个文件夹只取 5 个视频）
    python -m baseline.scripts.extract_ffpp_frames --folders original,Deepfakes --limit 5

    # 正式全量
    python -m baseline.scripts.extract_ffpp_frames --folders original,Deepfakes,Face2Face,FaceSwap,NeuralTextures,FaceShifter,DeepFakeDetection
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

# 默认的 YuNet 人脸检测模型（已随仓库放在 baseline/assets 下）
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "assets", "face_detection_yunet_2023mar.onnx")


def list_videos(folder: str):
    """列出文件夹下所有 mp4（按文件名排序，保证可复现）。"""
    names = [n for n in os.listdir(folder) if n.lower().endswith(".mp4")]
    return [os.path.join(folder, n) for n in sorted(names)]


def pick_largest_face(faces):
    """从 YuNet 的多个人脸框里挑面积最大的那个；返回 (x,y,w,h) 或 None。"""
    if faces is None or len(faces) == 0:
        return None
    best, best_area = None, -1.0
    for f in faces:
        x, y, w, h = f[:4]                       # YuNet 输出前 4 位是框
        area = float(w) * float(h)
        if area > best_area:
            best, best_area = (x, y, w, h), area
    return best


def square_crop_box(face_box, img_w, img_h, margin: float):
    """以人脸中心为基准，按 margin 倍外扩成正方形框；返回整数 (x0,y0,x1,y1)。"""
    x, y, w, h = face_box
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * margin                  # 正方形边长
    half = side / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    # 贴边处理：越界时整体平移回来，保证仍是正方形
    if x0 < 0: x1 += -x0; x0 = 0
    if y0 < 0: y1 += -y0; y0 = 0
    if x1 > img_w: x0 -= (x1 - img_w); x1 = img_w
    if y1 > img_h: y0 -= (y1 - img_h); y1 = img_h
    x0, y0 = max(0, x0), max(0, y0)
    return x0, y0, x1, y1


def extract_one(video_path, out_dir, frames: int, size: int, margin: float,
                model_path: str, jpeg_quality: int):
    """对单个视频抽帧 + 裁脸 + 存盘。返回 manifest 行列表。"""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[warn] 打不开视频，跳过: {video_path}", file=sys.stderr)
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))      # 视频总帧数
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or w <= 0 or h <= 0:
        cap.release()
        return []

    # 均匀取 frames 个帧位置（避开首尾各留一点余量）
    if frames == 1:
        targets = {total // 2}
    else:
        targets = {int(round(total * (i + 0.5) / frames)) for i in range(frames)}

    det = cv2.FaceDetectorYN.create(model_path, "", (w, h), 0.6, 0.3, 5000)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    idx, saved, last_idx = 0, 0, max(targets)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in targets:
            _, faces = det.detect(frame)
            box = pick_largest_face(faces)
            if box is not None:
                x0, y0, x1, y1 = square_crop_box(box, w, h, margin)
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    crop = frame
            else:
                # 没检出人脸就退化为整帧中心方形裁剪，保证不漏样本
                side = min(w, h)
                x0 = (w - side) // 2; y0 = (h - side) // 2
                crop = frame[y0:y0 + side, x0:x0 + side]
                box = (x0, y0, side, side)
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
            name = f"f{saved:03d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            rows.append({
                "folder": os.path.basename(os.path.dirname(video_path)),
                "stem": stem, "frame": saved, "frame_idx": idx,
                "video": os.path.basename(video_path),
                "face_x": int(box[0]), "face_y": int(box[1]),
                "face_w": int(box[2]), "face_h": int(box[3]),
                "image": os.path.join(stem, name),
            })
            saved += 1
        idx += 1
        if idx > last_idx:                     # 已过最后一个目标帧，可提前结束
            break

    cap.release()
    return rows


def _run_task(task):
    """进程池 worker：处理单个视频，返回 (folder, stem, manifest 行列表)。

    必须定义在模块顶层，否则不能被 pickle 传给子进程。
    """
    folder, vp, out_root, frames, size, margin, model, quality = task
    cv2.setNumThreads(1)          # 每个子进程单线程，避免多进程 × 多线程过度争抢
    stem = os.path.splitext(os.path.basename(vp))[0]
    rows = extract_one(vp, os.path.join(out_root, folder, stem),
                       frames, size, margin, model, quality)
    return folder, stem, rows


def main():
    p = argparse.ArgumentParser(description="FF++ 抽帧 + 人脸裁剪")
    p.add_argument("--src", default=r"D:\ffpp\FaceForensics++_C23", help="解压后的 FF++ 根目录")
    p.add_argument("--out", default=r"D:\ffpp_frames", help="输出根目录")
    p.add_argument("--folders", default="original,Deepfakes",
                   help="要处理的文件夹（逗号分隔）")
    p.add_argument("--frames", type=int, default=8, help="每个视频抽多少帧")
    p.add_argument("--size", type=int, default=256, help="裁剪后保存的边长")
    p.add_argument("--margin", type=float, default=1.8, help="人脸框外扩倍数")
    p.add_argument("--limit", type=int, default=0, help="每个文件夹只处理前 N 个视频（0=全部）")
    p.add_argument("--model", default=DEFAULT_MODEL, help="YuNet onnx 模型路径")
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=0, help="并行进程数（0=自动取 CPU 核数一半）")
    args = p.parse_args()

    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    workers = args.workers or max(1, (os.cpu_count() or 4) // 2)
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.csv")
    fields = ["folder", "stem", "frame", "frame_idx", "video",
              "face_x", "face_y", "face_w", "face_h", "image"]

    # 续跑：若 manifest 已存在则追加，并跳过已经抽过的 (folder, stem)
    done = set()
    file_exists = os.path.exists(manifest_path)
    if file_exists:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["folder"], r["stem"]))

    # 组装任务清单
    tasks = []
    for folder in folders:
        vdir = os.path.join(args.src, folder)
        if not os.path.isdir(vdir):
            print(f"[warn] 文件夹不存在，跳过: {vdir}", file=sys.stderr)
            continue
        vids = list_videos(vdir)
        if args.limit:
            vids = vids[:args.limit]
        todo = [vp for vp in vids
                if (folder, os.path.splitext(os.path.basename(vp))[0]) not in done]
        print(f"=== {folder}: {len(vids)} 个视频，待处理 {len(todo)} ===", flush=True)
        for vp in todo:
            tasks.append((folder, vp, args.out, args.frames, args.size,
                          args.margin, args.model, args.jpeg_quality))

    print(f"并行进程数: {workers} | 总任务: {len(tasks)}", flush=True)

    fout = open(manifest_path, "a" if file_exists else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=fields)
    if not file_exists:
        writer.writeheader()

    total, t0, last = 0, time.time(), time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_task, t) for t in tasks]
        for fut in as_completed(futures):
            folder, stem, rows = fut.result()
            for r in rows:
                writer.writerow(r)
            total += 1
            now = time.time()
            if total % 20 == 0 or now - last > 25:
                last = now
                fout.flush()
                el = now - t0
                print(f"  进度 {total}/{len(tasks)}  已用 {el:.0f}s  "
                      f"({el / max(total, 1):.2f}s/视频, 预计剩余 "
                      f"{(len(tasks) - total) * el / max(total, 1) / 60:.0f} 分钟)", flush=True)
    fout.close()
    print(f"完成：处理 {total} 个视频，manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
