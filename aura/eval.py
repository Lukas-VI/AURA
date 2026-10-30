"""检索评估（AURA 自建模型）：图库 = 测试集真实帧，查询 = 测试集伪造帧。

协议与 baseline/eval/retrieval.py 完全一致（命中 = 检索到 source_id），区别只在打分函数：
这里用【余弦相似度】对连续嵌入排序（对比学习模型的自然度量），同时给出【符号二值化 +
汉明距离】的读数，便于和零训练 pHash(64bit, 49.34%) 及 32×32 缩略图整图(58.80%) 直接对照。
"""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def encode_loader(model, loader, device):
    """把 loader 里所有图编码成 L2 归一化嵌入，并收集 id / 源视频 / source_id。"""
    model.eval()
    embs, ids, idents, srcs = [], [], [], []
    for batch in loader:
        z = model(batch["image"].to(device)).float()
        embs.append(F.normalize(z, dim=1).cpu())
        ids.extend(list(batch["id"]))
        idents.extend(list(batch["identity"]))
        srcs.extend(list(batch["source_id"]))
    return torch.cat(embs, dim=0), ids, idents, srcs


@torch.no_grad()
def _topk_cosine(query, db, k, chunk=256):
    """余弦相似度 Top-K（相似度越大越靠前）。分批算，避免物化 (Nq, Ndb) 大矩阵。"""
    query = F.normalize(query.float(), dim=1)
    db = F.normalize(db.float(), dim=1)
    out_d, out_i = [], []
    for s in range(0, query.shape[0], chunk):
        sim = query[s:s + chunk] @ db.t()
        td, ti = sim.topk(k, dim=1)
        out_d.append(td)
        out_i.append(ti)
    return torch.cat(out_d, dim=0), torch.cat(out_i, dim=0)


@torch.no_grad()
def _topk_hamming(query, db, k, chunk=256):
    """按符号二值化的哈希码算汉明距离（L1）Top-K（距离越小越靠前）。"""
    q = (query.float() > 0).float()
    d = (db.float() > 0).float()
    out_d, out_i = [], []
    for s in range(0, q.shape[0], chunk):
        dist = torch.cdist(q[s:s + chunk], d, p=1)
        td, ti = dist.topk(k, largest=False, dim=1)
        out_d.append(td)
        out_i.append(ti)
    return torch.cat(out_d, dim=0), torch.cat(out_i, dim=0)


def _score(indices, db_ids, q_srcs, q_idents, top_k):
    """命中判定：帧级 = 检索到 source_id；视频级 = 检索到同源视频（id 前缀 X）。"""
    frame = {k: 0 for k in top_k}
    video = {k: 0 for k in top_k}
    n = len(q_srcs)
    for i in range(n):
        retrieved = [db_ids[j] for j in indices[i].tolist()]
        sv = str(q_idents[i])
        for k in top_k:
            if q_srcs[i] in retrieved[:k]:
                frame[k] += 1
            if any(str(r).split("_f")[0] == sv for r in retrieved[:k]):
                video[k] += 1
    res = {}
    for k in top_k:
        res[f"frame_top{k}"] = frame[k] / max(n, 1)
        res[f"video_top{k}"] = video[k] / max(n, 1)
    res["n"] = n
    return res


@torch.no_grad()
def evaluate(model, db_loader, query_loader, device, top_k=(1, 5)):
    """返回余弦与汉明两套读数。"""
    db_z, db_ids, _, _ = encode_loader(model, db_loader, device)
    q_z, _, q_idents, q_srcs = encode_loader(model, query_loader, device)
    max_k = max(top_k)
    out = {"db": len(db_ids), "query": len(q_srcs)}
    for name, fn in (("cosine", _topk_cosine), ("hamming", _topk_hamming)):
        _, idx = fn(q_z, db_z, max_k)
        out[name] = _score(idx, db_ids, q_srcs, q_idents, top_k)
    return out


def format_report(res) -> str:
    lines = [f"图库 {res['db']} 张真实帧 / 查询 {res['query']} 张伪造帧"]
    for name in ("cosine", "hamming"):
        r = res[name]
        lines.append(f"  {name:7s}: " + " | ".join(
            f"{k}={r[k] * 100:.2f}%" for k in sorted(r) if k != "n"))
    return "\n".join(lines)


@torch.no_grad()
def evaluate_degradation(model, db_loader, query_samples, cfg, device,
                         top_k=(1, 5), levels=None):
    """退化鲁棒基准：图库【只编码一次且保持干净】，查询按六档确定性退化后分别检索。

    与 docs 冻结协议一致（只作用于查询）—— 真实取证场景里图库是干净的归档素材，待查的伪造图
    才经历社媒转存 / 二次压缩 / 裁剪。逐档重建 query dataset 与 loader；图库嵌入复用，避免重复前向。
    """
    from torch.utils.data import DataLoader
    from aura.data import ContrastiveDataset, DEGRADATION_LEVELS

    size = int(cfg["image_size"])
    eval_bs = int(cfg.get("eval_batch_size", 4))
    levels = list(levels) if levels else list(DEGRADATION_LEVELS)

    db_z, db_ids, _, _ = encode_loader(model, db_loader, device)
    max_k = max(top_k)
    out = {"db": len(db_ids), "levels": {}}
    for lvl in levels:
        q_ds = ContrastiveDataset(query_samples, image_size=size, train=False,
                                  seed=int(cfg.get("seed", 42)), degrade_level=lvl)
        q_loader = DataLoader(q_ds, batch_size=eval_bs, shuffle=False, num_workers=0)
        q_z, _, q_idents, q_srcs = encode_loader(model, q_loader, device)
        entry = {"query": len(q_srcs)}
        for name, fn in (("cosine", _topk_cosine), ("hamming", _topk_hamming)):
            _, idx = fn(q_z, db_z, max_k)
            entry[name] = _score(idx, db_ids, q_srcs, q_idents, top_k)
        out["levels"][lvl] = entry
    return out


def format_degradation(res) -> str:
    lines = [f"图库 {res['db']} 张真实帧（干净） / 查询 {res['levels'][next(iter(res['levels']))]['query']} 张伪造帧（逐档退化）"]
    lines.append(f"  {'档位':>9s} | {'cos帧T1':>8s} {'cos帧T5':>8s} | {'ham帧T1':>8s} {'ham帧T5':>8s}")
    for lvl, e in res["levels"].items():
        c, hh = e["cosine"], e["hamming"]
        lines.append(f"  {lvl:>9s} | {c['frame_top1'] * 100:7.2f}% {c['frame_top5'] * 100:7.2f}% | "
                     f"{hh['frame_top1'] * 100:7.2f}% {hh['frame_top5'] * 100:7.2f}%")
    return "\n".join(lines)


def _auc_rank(scores, labels) -> float:
    """AUC = Mann-Whitney U / (n+·n−)，用平均秩处理并列（与 sklearn 口径一致，无第三方依赖）。

    scores 越大越像正样本；labels: 1 = 闭集命中（正）、0 = 开集拒识（负）。
    """
    uniq, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg_rank = csum - (cnt - 1) / 2.0                       # 每个并列组的平均秩（1-based）
    ranks = avg_rank[inv]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def evaluate_openset(model, db_loader, closed_samples, open_samples, cfg, device,
                     top_k=(1, 5), fpr=0.01):
    """开集拒识协议（HashVAE 与两家免训练基线都没有的指标）。

    动机：现有协议是【闭集】的 —— 查询的源图一定在图库里，所以"检索到 source_id"永远可以
    命中；但这假设了"待查伪造图必然来自已归档的真实视频"。真实取证场景里恰恰相反：多数待查
    图片的源头【根本不在库中】，系统必须先判"这条查询库里没有归宿"，再谈检索 —— 这就是开集
    拒识（open-set / out-of-distribution rejection）。

    口径
    ----
    图库    = 闭集视频（如 150/200 条）的真实帧，干净、只编码一次。
    闭集查询 = 源视频落在闭集内的伪造帧 —— 库里【有】归宿，应当正常命中且被接受。
    开集查询 = 源视频被整条排除（如另 50 条）的伪造帧 —— 库里【无】归宿，应当被拒绝。
    闭集块帧级/视频级 Top-K 应与标准协议同量级（图库略小，Top-1 会略偏高，属预期）。
    拒识打分 = 查询对全库的 top-1 余弦相似度（越大越像库内成员）；闭集标 1、开集标 0，报
    AUC 与 TPR@FPR（阈值在【开集分布】的目标 FPR 分位上标定，模拟"用已知未知样本定阈值"）。
    """
    from torch.utils.data import DataLoader
    from aura.data import ContrastiveDataset

    size = int(cfg["image_size"])
    eval_bs = int(cfg.get("eval_batch_size", 4))
    top_k = tuple(sorted(int(k) for k in top_k))
    max_k = max(top_k)

    db_z, db_ids, _, _ = encode_loader(model, db_loader, device)
    out = {"db": len(db_ids), "top_k": list(top_k), "fpr_target": float(fpr)}

    t1_sim = {}
    for name, samples in (("closed", closed_samples), ("open", open_samples)):
        ds = ContrastiveDataset(samples, image_size=size, train=False,
                                seed=int(cfg.get("seed", 42)))
        loader = DataLoader(ds, batch_size=eval_bs, shuffle=False, num_workers=0)
        z, _, idents, srcs = encode_loader(model, loader, device)
        d, idx = _topk_cosine(z, db_z, max_k)
        rec = {"query": len(srcs)}
        rec.update(_score(idx, db_ids, srcs, idents, top_k))
        out[name] = rec
        t1_sim[name] = d[:, 0].cpu().numpy().astype(np.float64)      # top-1 相似度即拒识打分

    cs, os_ = t1_sim["closed"], t1_sim["open"]
    labels = np.concatenate([np.ones(len(cs)), np.zeros(len(os_))])
    scores = np.concatenate([cs, os_])
    thr = float(np.percentile(os_, 100.0 * (1.0 - float(fpr)))) if len(os_) else float("nan")
    out["reject"] = {
        "score": "top1_cosine",
        "auc": _auc_rank(scores, labels),
        "threshold": thr,
        "fpr_target": float(fpr),
        "fpr_real": float((os_ >= thr).mean()) if len(os_) else float("nan"),
        "tpr_at_fpr": float((cs >= thr).mean()) if len(cs) else float("nan"),
        "fnr_at_fpr": float((cs < thr).mean()) if len(cs) else float("nan"),
        "closed_sim_mean": float(cs.mean()) if len(cs) else float("nan"),
        "open_sim_mean": float(os_.mean()) if len(os_) else float("nan"),
        "n_closed": len(cs), "n_open": len(os_),
    }
    return out


def format_openset(res) -> str:
    c, o, r = res["closed"], res["open"], res["reject"]
    lines = [f"开集拒识：图库 {res['db']} 张真实帧（闭集视频） | "
             f"闭集查询 {c['query']} / 开集查询 {o['query']} 张伪造帧"]
    for tag, e in (("闭集（应正常命中）", c), ("开集（源头不在库，应≈0）", o)):
        ks = [k for k in sorted(e) if k.startswith(("frame_", "video_"))]
        lines.append(f"  {tag}: " + " | ".join(f"{k}={e[k] * 100:.2f}%" for k in ks))
    lines.append(f"  拒识（top-1 余弦）: AUC {r['auc'] * 100:.2f}% | "
                 f"TPR@FPR={r['fpr_target'] * 100:.0f}% = {r['tpr_at_fpr'] * 100:.2f}% "
                 f"(实际 FPR {r['fpr_real'] * 100:.2f}%) | 阈值 {r['threshold']:.4f}")
    lines.append(f"  相似度均值: 闭集 {r['closed_sim_mean']:.4f} / 开集 {r['open_sim_mean']:.4f}"
                 f"（差 {r['closed_sim_mean'] - r['open_sim_mean']:.4f}）")
    return "\n".join(lines)


@torch.no_grad()
def evaluate_openset_levels(model, db_loader, closed_samples, open_samples, cfg, device,
                            top_k=(1, 5), fpr=0.01, levels=None):
    """开集拒识 × 退化协议：图库保持【干净且只编码一次】，逐档对闭集/开集查询【同时】退化。

    动机（承接 §3.14）：干净档下闭集查询 top-1 相似度 0.989、开集 0.717，拒识 AUC≈99.7% 近乎
    平凡 —— 因为干净伪造帧的背景与源图逐像素几乎相同，库内/库外的可分性被"像素级近重复"掩盖。
    真正有意义的测评要在退化档下做：JPEG / 裁剪 / 旋转把查询相似度整体压低后，"这条查询库里
    到底有没有归宿"才见真章。每档同时报 (a) 闭集块帧级/视频级 Top-K（应有明显回落），
    (b) 以 top-1 余弦为判据的拒识 AUC / TPR@FPR / 闭集-开集相似度均值。
    """
    from torch.utils.data import DataLoader
    from aura.data import ContrastiveDataset, DEGRADATION_LEVELS, GEOM_LEVELS

    size = int(cfg["image_size"])
    eval_bs = int(cfg.get("eval_batch_size", 4))
    top_k = tuple(sorted(int(k) for k in top_k))
    max_k = max(top_k)
    if levels is None:
        # 光度六档 + 几何六档（clean 只出现一次）= 12 档
        levels = list(DEGRADATION_LEVELS) + [g for g in GEOM_LEVELS if g not in DEGRADATION_LEVELS]
    levels = list(levels)

    db_z, db_ids, _, _ = encode_loader(model, db_loader, device)
    out = {"db": len(db_ids), "top_k": list(top_k), "fpr_target": float(fpr), "levels": {}}
    for lvl in levels:
        entry = {}
        sims = {}
        for name, samples in (("closed", closed_samples), ("open", open_samples)):
            ds = ContrastiveDataset(samples, image_size=size, train=False,
                                    seed=int(cfg.get("seed", 42)), degrade_level=lvl)
            loader = DataLoader(ds, batch_size=eval_bs, shuffle=False, num_workers=0)
            z, _, idents, srcs = encode_loader(model, loader, device)
            d, idx = _topk_cosine(z, db_z, max_k)
            rec = {"query": len(srcs)}
            rec.update(_score(idx, db_ids, srcs, idents, top_k))
            entry[name] = rec
            sims[name] = d[:, 0].cpu().numpy().astype(np.float64)

        cs, os_ = sims["closed"], sims["open"]
        labels = np.concatenate([np.ones(len(cs)), np.zeros(len(os_))])
        scores = np.concatenate([cs, os_])
        thr = float(np.percentile(os_, 100.0 * (1.0 - float(fpr)))) if len(os_) else float("nan")
        entry["reject"] = {
            "score": "top1_cosine",
            "auc": _auc_rank(scores, labels),
            "threshold": thr,
            "fpr_target": float(fpr),
            "fpr_real": float((os_ >= thr).mean()) if len(os_) else float("nan"),
            "tpr_at_fpr": float((cs >= thr).mean()) if len(cs) else float("nan"),
            "fnr_at_fpr": float((cs < thr).mean()) if len(cs) else float("nan"),
            "closed_sim_mean": float(cs.mean()) if len(cs) else float("nan"),
            "open_sim_mean": float(os_.mean()) if len(os_) else float("nan"),
            "n_closed": len(cs), "n_open": len(os_),
        }
        out["levels"][lvl] = entry
    return out


def format_openset_levels(res) -> str:
    lv = res["levels"]
    first = lv[next(iter(lv))]
    lines = [f"开集拒识 × 退化：图库 {res['db']} 张真实帧（干净、只编码一次） | "
             f"闭集查询 {first['closed']['query']} / 开集查询 {first['open']['query']} 张伪造帧（逐档退化）"]
    lines.append(f"  {'档位':>9s} | {'闭集帧T1':>8s} {'闭集帧T5':>8s} | {'拒识AUC':>8s} {'TPR@1%':>7s} | "
                 f"{'闭集/开集均值':>15s} | {'开集帧T1':>7s}")
    for lvl, e in lv.items():
        c, o, r = e["closed"], e["open"], e["reject"]
        lines.append(f"  {lvl:>9s} | {c['frame_top1'] * 100:7.2f}% {c['frame_top5'] * 100:7.2f}% | "
                     f"{r['auc'] * 100:7.2f}% {r['tpr_at_fpr'] * 100:6.2f}% | "
                     f"{r['closed_sim_mean']:.3f}/{r['open_sim_mean']:.3f} | "
                     f"{o['frame_top1'] * 100:6.2f}%")
    return "\n".join(lines)
