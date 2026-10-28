"""coview command line.

    python -m coview.cli synth --out data/synth
    python -m coview.cli train --data data/synth --encoder handcrafted     # offline smoke run
    python -m coview.cli train --data data/microlens --encoder clip        # the real run (downloads CLIP once)

Outputs (in --out): head.pt, item_embeddings.npz (item id + fused 128-d vector, the
file ReelFunnel's --content-emb flag reads), metrics.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import SynthConfig, co_engagement_pairs, cold_start_split, cover_path, load_interactions, load_titles, \
    make_synthetic
from .encoders import cached_features
from .model import MODES, CoViewHead, embed, train_head


def _prep(x: np.ndarray, standardize: bool) -> np.ndarray:
    x = x.astype(np.float32)
    if standardize:  # handcrafted features live on different scales
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def _recall_ndcg(scores: np.ndarray, cand_items: np.ndarray, gts: list[set], k: int):
    top = np.argsort(-scores, axis=1)[:, :k]
    rec, ndcg = [], []
    for row, gt in zip(top, gts):
        hits = [cand_items[j] in gt for j in row]
        rec.append(sum(hits) / min(len(gt), k))
        dcg = sum(1 / np.log2(i + 2) for i, h in enumerate(hits) if h)
        ndcg.append(dcg / sum(1 / np.log2(i + 2) for i in range(min(len(gt), k))))
    return float(np.mean(rec)), float(np.mean(ndcg))


def evaluate(split, row_of, img, txt, head, hist_len: int = 20) -> dict:
    """Cold-start retrieval: rank never-seen items for each user from their training history."""
    cold = split.cold_items
    cold_set = set(cold.tolist())
    test_cold = split.test[split.test.item.isin(cold_set)]
    gts = test_cold.groupby("user").item.apply(set)
    hist = split.train.groupby("user").item.apply(lambda s: s.to_numpy()[-hist_len:])
    users = [u for u in gts.index if u in hist.index]
    gt_list = [gts[u] for u in users]
    cold_rows = np.array([row_of[i] for i in cold])

    def query(item_emb):
        q = np.stack([item_emb[[row_of[i] for i in hist[u]]].mean(0) for u in users])
        return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

    out = {"cold_items": int(len(cold)), "eval_users": len(users),
           "cold_interactions": int(len(test_cold))}
    for k in (10, 50):
        out[f"chance_recall@{k}"] = round(k / len(cold), 4)

    zero_shot = np.concatenate([img, txt], 1) / np.sqrt(2)
    q0 = query(zero_shot)
    for k in (10, 50):
        r, n = _recall_ndcg(q0 @ zero_shot[cold_rows].T, cold, gt_list, k)
        out[f"zero_shot_frozen_encoder_recall@{k}"] = round(r, 4)
    out["zero_shot_frozen_encoder_ndcg@10"] = round(_recall_ndcg(q0 @ zero_shot[cold_rows].T, cold, gt_list, 10)[1], 4)

    fused = embed(head, img, txt, "fused")
    q = query(fused)  # history items are warm and have both modalities
    for mode in MODES:
        e = embed(head, img[cold_rows], txt[cold_rows], mode)
        for k in (10, 50):
            r, n = _recall_ndcg(q @ e.T, cold, gt_list, k)
            out[f"coview_{mode}_recall@{k}"] = round(r, 4)
            if k == 10:
                out[f"coview_{mode}_ndcg@10"] = round(n, 4)
    out["lift_fused_vs_zero_shot_recall@50"] = \
        f"{out['coview_fused_recall@50'] / max(out['zero_shot_frozen_encoder_recall@50'], 1e-9) - 1:+.1%}"
    out["lift_fused_vs_chance_recall@50"] = f"{out['coview_fused_recall@50'] / out['chance_recall@50']:.1f}x"
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="coview")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--users", type=int, default=3000)
    s.add_argument("--items", type=int, default=3000)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="folder with pairs.csv, titles.csv, covers/")
    t.add_argument("--pairs"); t.add_argument("--titles"); t.add_argument("--covers")
    t.add_argument("--encoder", default="clip", help="clip | siglip | handcrafted | any HF model id")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--window", type=int, default=5)
    t.add_argument("--test-frac", type=float, default=0.15)
    t.add_argument("--out", default="artifacts")
    a = p.parse_args(argv)

    if a.cmd == "synth":
        make_synthetic(a.out, SynthConfig(n_users=a.users, n_items=a.items))
        print(f"wrote {a.out}")
        return

    d = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load_interactions(a.pairs or d / "pairs.csv")
    titles = load_titles(a.titles or d / "titles.csv")
    covers = a.covers or d / "covers"
    items = np.sort(df.item.unique())
    row_of = {int(it): i for i, it in enumerate(items)}
    img, txt = cached_features(out / "cache", items, titles, lambda i: cover_path(covers, i), a.encoder)
    std = a.encoder == "handcrafted"
    img, txt = _prep(img, std), _prep(txt, std)
    print(f"features: image {img.shape}, text {txt.shape} ({time.time() - t0:.0f}s)", flush=True)

    split = cold_start_split(df, a.test_frac)
    pairs = co_engagement_pairs(split.train, a.window)
    rows = np.vectorize(row_of.get)(pairs)
    print(f"{len(df):,} interactions, {len(split.warm_items):,} warm / {len(split.cold_items):,} cold items, "
          f"{len(pairs):,} co-engagement pairs", flush=True)

    head = CoViewHead(img.shape[1], txt.shape[1])
    t1 = time.time()
    train_head(head, img, txt, rows, epochs=a.epochs)
    train_s = time.time() - t1

    metrics = evaluate(split, row_of, img, txt, head)
    metrics.update({"encoder": a.encoder, "train_pairs": int(len(pairs)), "train_seconds": round(train_s, 1)})
    print(json.dumps(metrics, indent=2))
    torch.save(head.state_dict(), out / "head.pt")
    np.savez_compressed(out / "item_embeddings.npz", video_id=items, emb=embed(head, img, txt, "fused"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
