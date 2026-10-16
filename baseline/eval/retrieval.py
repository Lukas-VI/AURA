"""检索评估：数据库为真实图，查询为伪造图，命中条件为“源图 id 落在 Top-K 内”。

论文出处：Liu et al. 2026, 3.5 Deepfake forgery retrieval（Algorithm 1）。

本项目对应的场景是论文 Algorithm 1 的“情形(1)：前景被篡改”（伪造图由换脸得到，
背景保持源图不变）。此时检索策略的第一步（Step 3）就是：
    “Compute the hamming distance among the C_b with each background hash code,
     and keep the minimum M1 results in candidates pool D1.”
即【用背景哈希码 C_b 做汉明距离检索】得到候选池 D1 —— 该候选池就是源图。
这与 AURA 的核心假设一致：背景是伪造图可靠的身份指纹。

因此这里默认用【背景哈希码】作为检索键（key='bg'）；同时保留 key='full'
（前后景拼接码）与 key='fg'（前景码）以便对照。
"""
import torch

# 三种检索键 -> 模型 forward 输出里的字段名
_KEY_FIELD = {"bg": "code_b", "fg": "code_f", "full": "code"}


@torch.no_grad()
def build_database(model, loader, device, key: str = "bg"):
    """把数据库（真实图）编码成哈希码，并记录对应的 id。"""
    model.eval()
    field = _KEY_FIELD[key]
    hashes, ids = [], []
    for batch in loader:
        img = batch["image"].to(device)
        mask = batch.get("mask")                     # 逐样本前景掩膜（诊断用的临时 loader 可能没有）
        out = model(img, mask=None if mask is None else mask.to(device),
                    need_recon=False, need_pred=False)   # 推理只需哈希码
        hashes.append(out[field].cpu())
        ids.extend(list(batch["id"]))
    return torch.cat(hashes, dim=0), ids


@torch.no_grad()
def evaluate_retrieval(model, database_loader, query_loader, device, top_k=(1, 5), key: str = "bg"):
    """计算检索命中率。

    参数
    ----
    key : 'bg'（背景码，默认，论文情形1的第一步）/ 'fg'（前景码）/ 'full'（拼接码）

    返回 dict：{'top1': .., 'top5': .., 'n': ..}
    命中判定：查询样本的 source_id 是否出现在检索到的 Top-K 数据库 id 里。
    """
    model.eval()
    max_k = max(top_k)
    field = _KEY_FIELD[key]

    db_hashes, db_ids = build_database(model, database_loader, device, key=key)

    correct = {k: 0 for k in top_k}
    total = 0
    for batch in query_loader:
        img = batch["image"].to(device)
        mask = batch.get("mask")
        out = model(img, mask=None if mask is None else mask.to(device),
                    need_recon=False, need_pred=False)
        hash_code = out[field].cpu()
        src_ids = list(batch["source_id"])

        # 汉明距离：硬量化后 L1(cdist p=1) 等价于汉明距离
        distances = torch.cdist(hash_code.float(), db_hashes.float(), p=1)
        _, indices = distances.topk(max_k, largest=False)   # 距离最小 = 最相似

        for i in range(len(src_ids)):
            retrieved = [db_ids[j] for j in indices[i].tolist()]
            total += 1
            for k in top_k:
                if src_ids[i] in retrieved[:k]:
                    correct[k] += 1

    result = {f"top{k}": correct[k] / max(total, 1) for k in top_k}
    result["n"] = total
    return result
