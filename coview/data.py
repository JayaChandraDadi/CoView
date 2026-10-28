"""MicroLens loader and a synthetic generator with the same file layout.

MicroLens is a short-video recommendation dataset that ships raw content: a
cover image and a title per item, plus user-item interactions with timestamps.
That makes it the right public stand-in for content-based cold start.

Layout the loader expects (paths are configurable on the CLI):
    pairs.csv     user, item, timestamp
    titles.csv    item, title
    covers/       <item>.jpg
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def load_interactions(pairs: str | Path) -> pd.DataFrame:
    df = pd.read_csv(pairs, sep=None, engine="python")
    df = df.iloc[:, :3]
    df.columns = ["user", "item", "timestamp"]
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_titles(path: str | Path) -> dict[int, str]:
    t = pd.read_csv(path, sep=None, engine="python")
    t = t.iloc[:, :2]
    t.columns = ["item", "title"]
    return dict(zip(t.item.astype(int), t.title.fillna("").astype(str)))


def cover_path(covers_dir: str | Path, item: int) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path(covers_dir) / f"{item}{ext}"
        if p.exists():
            return p
    return Path(covers_dir) / f"{item}.jpg"


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    warm_items: np.ndarray
    cold_items: np.ndarray


def cold_start_split(df: pd.DataFrame, test_frac: float = 0.15) -> Split:
    """Time split; items whose first interaction falls in the test window are cold.

    Cold items have zero training interactions, so an ID embedding cannot have
    learned anything about them. Only content can place them.
    """
    cut = df.timestamp.quantile(1 - test_frac)
    train, test = df[df.timestamp < cut], df[df.timestamp >= cut]
    warm = np.unique(train.item)
    cold = np.setdiff1d(np.unique(test.item), warm)
    return Split(train, test, warm, cold)


def co_engagement_pairs(train: pd.DataFrame, window: int = 5, min_count: int = 1, max_pairs: int = 2_000_000,
                        seed: int = 0) -> np.ndarray:
    """(item_a, item_b) pairs engaged by the same user within `window` consecutive interactions.

    This is the behavioural signal the content embeddings are aligned to: items people
    engage with together, rather than items that merely look alike.
    """
    counts: dict[tuple[int, int], int] = {}
    for _, items in train.groupby("user").item:
        seq = items.to_numpy()
        for i in range(len(seq)):
            for j in range(i + 1, min(i + 1 + window, len(seq))):
                a, b = int(seq[i]), int(seq[j])
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                counts[key] = counts.get(key, 0) + 1
    pairs = np.array([k for k, c in counts.items() if c >= min_count], dtype=np.int64)
    if len(pairs) > max_pairs:
        pairs = pairs[np.random.default_rng(seed).choice(len(pairs), max_pairs, replace=False)]
    return pairs


# ---------------------------------------------------------------------- synthetic
@dataclass
class SynthConfig:
    n_users: int = 3000
    n_items: int = 3000
    n_topics: int = 24
    interactions_per_user: int = 40
    img: int = 64
    seed: int = 5


def make_synthetic(out_dir: str | Path, cfg: SynthConfig = SynthConfig()) -> None:
    """Items have a hidden topic. Its cover colour/shape and title words carry the topic noisily,
    and users engage by topic, so content genuinely predicts co-engagement."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(cfg.seed)
    out = Path(out_dir)
    (out / "covers").mkdir(parents=True, exist_ok=True)
    topic = rng.integers(0, cfg.n_topics, cfg.n_items)
    hues = rng.random(cfg.n_topics)
    shapes = rng.integers(0, 3, cfg.n_topics)
    vocab = [[f"w{t}_{k}" for k in range(12)] for t in range(cfg.n_topics)]
    common = ["the", "best", "my", "day", "new", "wow", "watch", "this", "fun", "video", "today", "look"]

    import colorsys
    titles = []
    for i in range(cfg.n_items):
        t = topic[i]
        noisy_t = t if rng.random() < 0.75 else rng.integers(0, cfg.n_topics)  # covers mislead 25% of the time
        r, g, b = (int(255 * x) for x in colorsys.hsv_to_rgb((hues[noisy_t] + rng.normal(0, 0.03)) % 1, 0.7, 0.9))
        im = Image.new("RGB", (cfg.img, cfg.img), tuple(int(v) for v in rng.integers(0, 60, 3)))
        dr = ImageDraw.Draw(im)
        x0, y0 = rng.integers(4, 20, 2)
        x1, y1 = x0 + rng.integers(24, 40), y0 + rng.integers(24, 40)
        [dr.ellipse, dr.rectangle, lambda box, fill: dr.polygon([(box[0], box[3]), ((box[0] + box[2]) // 2, box[1]),
                                                                 (box[2], box[3])], fill=fill)][shapes[noisy_t]](
            (x0, y0, x1, y1), fill=(r, g, b))
        arr = np.asarray(im).astype(np.int16) + rng.integers(-12, 12, (cfg.img, cfg.img, 3))
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(out / "covers" / f"{i + 1}.jpg", quality=90)
        words = [rng.choice(vocab[t]) if rng.random() < 0.55 else rng.choice(common) for _ in range(rng.integers(3, 8))]
        titles.append((i + 1, " ".join(words)))
    pd.DataFrame(titles, columns=["item", "title"]).to_csv(out / "titles.csv", index=False)

    # Items are released over time; users draw items from their favourite topics among released ones.
    release = rng.random(cfg.n_items)
    by_topic = {t: np.flatnonzero(topic == t) for t in range(cfg.n_topics)}
    rows = []
    for u in range(cfg.n_users):
        favs = rng.choice(cfg.n_topics, rng.integers(1, 4), replace=False)
        times = np.sort(rng.random(cfg.interactions_per_user))
        for t in times:
            tp = rng.choice(favs) if rng.random() < 0.85 else rng.integers(0, cfg.n_topics)
            cands = by_topic[tp][release[by_topic[tp]] <= t]
            if len(cands) == 0:
                continue
            # Newer items are favoured, so recently released (cold) items get real test traffic.
            w = np.exp((release[cands] - t) * 8)
            rows.append((u + 1, int(rng.choice(cands, p=w / w.sum())) + 1, int(t * 1e9)))
    pd.DataFrame(rows, columns=["user", "item", "timestamp"]).to_csv(out / "pairs.csv", index=False)
