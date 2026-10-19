"""检索评估（AURA 自建模型）：图库 = 测试集真实帧，查询 = 测试集伪造帧。

协议与 baseline/eval/retrieval.py 完全一致（命中 = 检索到 source_id），区别只在打分函数：
这里用【余弦相似度】对连续嵌入排序（对比学习模型的自然度量），同时给出【符号二值化 +
汉明距离】的读数，便于和零训练 pHash(64bit, 49.34%) 及 32×32 缩略图整图(58.80%) 直接对照。
"""
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
