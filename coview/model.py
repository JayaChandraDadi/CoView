"""Fusion head trained with symmetric InfoNCE on co-engagement pairs.

Image and text features each pass through a small MLP; a learned gate mixes the
two into one 128-d L2-normalised item vector. During training one modality is
dropped at random (never both), so a single checkpoint can embed an item from
its cover alone, its title alone, or both, which is what cold-start traffic
actually looks like (missing covers, empty titles).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("fused", "image", "text")


def _branch(d_in, d_out):
    return nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, d_out))


class CoViewHead(nn.Module):
    def __init__(self, d_img: int, d_txt: int, dim: int = 128):
        super().__init__()
        self.img = _branch(d_img, dim)
        self.txt = _branch(d_txt, dim)
        self.gate = nn.Linear(2 * dim, 2)
        self.log_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # CLIP-style learnable temperature

    def forward(self, img, txt, use_img, use_txt):
        """`use_img` / `use_txt` are [B] bool masks of which modality is present."""
        hi = self.img(img) * use_img.unsqueeze(-1)
        ht = self.txt(txt) * use_txt.unsqueeze(-1)
        logits = self.gate(torch.cat([hi, ht], -1))
        logits = logits.masked_fill(torch.stack([~use_img, ~use_txt], -1), -1e4)
        w = torch.softmax(logits, -1)
        return F.normalize(w[:, :1] * hi + w[:, 1:] * ht, dim=-1)


def masks(n: int, mode: str):
    on, off = torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    return {"fused": (on, on), "image": (on, off), "text": (off, on)}[mode]


def info_nce(za, zb, ids_a, ids_b, log_temp):
    """Symmetric in-batch InfoNCE; other copies of the same item are not counted as negatives."""
    scale = log_temp.exp().clamp(max=100)
    logits = za @ zb.T * scale
    dup = (ids_a.unsqueeze(1) == ids_b.unsqueeze(0)) & ~torch.eye(len(ids_a), dtype=torch.bool)
    logits = logits.masked_fill(dup, -1e4)
    y = torch.arange(len(za))
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def train_head(head: CoViewHead, img: np.ndarray, txt: np.ndarray, pairs_rows: np.ndarray,
               epochs: int = 8, batch: int = 512, lr: float = 1e-3, modality_dropout: float = 0.3, seed: int = 0):
    """`pairs_rows` are (row_a, row_b) indices into the feature matrices."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    I, T = torch.as_tensor(img), torch.as_tensor(txt)
    P = torch.as_tensor(pairs_rows)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(P) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    head.train()
    for ep in range(epochs):
        perm = P[torch.randperm(len(P), generator=g)]
        tot = 0.0
        for s in range(0, len(perm), batch):
            b = perm[s: s + batch]
            if len(b) < 8:
                continue
            ra, rb = b[:, 0], b[:, 1]

            def side(rows):
                n = len(rows)
                drop = torch.rand(n, generator=g) < modality_dropout
                which = torch.rand(n, generator=g) < 0.5
                use_i, use_t = ~(drop & which), ~(drop & ~which)
                return head(I[rows], T[rows], use_i, use_t)

            loss = info_nce(side(ra), side(rb), ra, rb, head.log_temp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
        print(f"  epoch {ep + 1}/{epochs}: InfoNCE {tot / len(P):.4f}  temp {1 / head.log_temp.exp().item():.3f}",
              flush=True)
    head.eval()


@torch.no_grad()
def embed(head: CoViewHead, img: np.ndarray, txt: np.ndarray, mode: str = "fused") -> np.ndarray:
    I, T = torch.as_tensor(img), torch.as_tensor(txt)
    ui, ut = masks(len(I), mode)
    return head(I, T, ui, ut).numpy()
