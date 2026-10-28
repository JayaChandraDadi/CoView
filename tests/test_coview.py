from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coview.data import SynthConfig, co_engagement_pairs, cold_start_split, make_synthetic  # noqa: E402
from coview.model import CoViewHead, embed, info_nce, masks  # noqa: E402


def test_cold_items_have_no_training_interactions():
    df = pd.DataFrame({"user": [1, 1, 2, 2, 3], "item": [10, 11, 10, 12, 13], "timestamp": [1, 2, 3, 9, 10]})
    s = cold_start_split(df, test_frac=0.4)
    assert set(s.cold_items) & set(s.train.item) == set()
    assert 13 in set(s.cold_items)


def test_pairs_are_windowed_and_canonical():
    df = pd.DataFrame({"user": [1] * 4, "item": [5, 3, 5, 9], "timestamp": range(4)})
    pairs = {tuple(p) for p in co_engagement_pairs(df, window=1)}
    assert pairs == {(3, 5), (5, 9)}  # (5,5) self pair dropped; window 1 = neighbours only


def test_single_modality_modes_ignore_the_other_input():
    head = CoViewHead(8, 6, dim=16).eval()
    img, txt = np.random.randn(4, 8).astype(np.float32), np.random.randn(4, 6).astype(np.float32)
    e_img = embed(head, img, txt, "image")
    e_img2 = embed(head, img, np.random.randn(4, 6).astype(np.float32), "image")
    np.testing.assert_allclose(e_img, e_img2, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(embed(head, img, txt, "fused"), axis=1), 1, atol=1e-5)


def test_info_nce_masks_duplicate_items():
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    ids = torch.tensor([1, 1, 2, 3])
    loss_dup = info_nce(z, z, ids, ids, torch.tensor(np.log(10.0)))
    assert torch.isfinite(loss_dup)


def test_synthetic_layout(tmp_path):
    make_synthetic(tmp_path, SynthConfig(n_users=30, n_items=40, interactions_per_user=10, img=32))
    assert (tmp_path / "pairs.csv").exists() and (tmp_path / "titles.csv").exists()
    assert len(list((tmp_path / "covers").glob("*.jpg"))) == 40
    assert masks(3, "text")[0].sum() == 0
