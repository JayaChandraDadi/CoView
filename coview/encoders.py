"""Frozen content encoders that turn covers and titles into feature vectors.

Default: a pretrained CLIP or SigLIP checkpoint from Hugging Face, run once over
every item and cached to disk, so training only touches the small fusion head
and fits comfortably on an M-series laptop (MPS) or CPU.

`handcrafted` is an offline fallback for tests and air-gapped machines: colour
histograms and coarse layout for images, hashed bag-of-words for titles. It is
not a pretrained model and the README labels its numbers that way.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

HF_MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
}


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def hf_features(items, titles: dict, cover, model_name: str = "clip", batch: int = 64):
    """(image_features, text_features) from a frozen Hugging Face vision-language model."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    name = HF_MODELS.get(model_name, model_name)
    dev = _device()
    model = AutoModel.from_pretrained(name).to(dev).eval()
    proc = AutoProcessor.from_pretrained(name)
    img_out, txt_out = [], []
    with torch.no_grad():
        for s in range(0, len(items), batch):
            chunk = items[s: s + batch]
            ims = []
            for i in chunk:
                try:
                    ims.append(Image.open(cover(i)).convert("RGB"))
                except FileNotFoundError:
                    ims.append(Image.new("RGB", (224, 224)))
            txt = [titles.get(int(i), "") or " " for i in chunk]
            pi = proc(images=ims, return_tensors="pt").to(dev)
            pt = proc(text=txt, return_tensors="pt", padding="max_length", truncation=True,
                      max_length=64 if "siglip" in name else 77).to(dev)
            img_out.append(model.get_image_features(**pi).float().cpu().numpy())
            txt_out.append(model.get_text_features(**pt).float().cpu().numpy())
            if s % (batch * 20) == 0:
                print(f"  encoded {s + len(chunk)}/{len(items)} items", flush=True)
    return np.concatenate(img_out), np.concatenate(txt_out)


def _hash(word: str, dim: int) -> int:
    return int(hashlib.md5(word.encode()).hexdigest(), 16) % dim


def handcrafted_features(items, titles: dict, cover, text_dim: int = 512):
    """Offline fallback: no download, no pretrained weights."""
    from PIL import Image

    img, txt = [], []
    for i in items:
        try:
            im = Image.open(cover(i)).convert("RGB").resize((32, 32))
            a = np.asarray(im, np.float32) / 255.0
        except FileNotFoundError:
            a = np.zeros((32, 32, 3), np.float32)
        hsv = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).convert("HSV"), np.float32) / 255.0
        hue_hist = np.histogram(hsv[..., 0], bins=24, range=(0, 1), weights=hsv[..., 1])[0]
        rgb_hist = np.concatenate([np.histogram(a[..., c], bins=8, range=(0, 1))[0] for c in range(3)])
        layout = a.reshape(4, 8, 4, 8, 3).mean((1, 3)).ravel()          # 4x4 grid of mean colours
        mask = (hsv[..., 1] > 0.4).astype(np.float32)                     # rough shape silhouette
        shape = mask.reshape(8, 4, 8, 4).mean((1, 3)).ravel()
        img.append(np.concatenate([hue_hist / (hue_hist.sum() + 1e-6), rgb_hist / 1024.0, layout, shape]))
        v = np.zeros(text_dim, np.float32)
        for w in re.findall(r"\w+", titles.get(int(i), "").lower()):
            v[_hash(w, text_dim)] += 1.0
        txt.append(v / (np.linalg.norm(v) + 1e-6))
    return np.asarray(img, np.float32), np.asarray(txt, np.float32)


def cached_features(cache: Path, items, titles, cover, encoder: str):
    cache = Path(cache)
    f = cache / f"features_{encoder.replace('/', '_')}.npz"
    if f.exists():
        z = np.load(f)
        if np.array_equal(z["items"], items):
            return z["img"], z["txt"]
    cache.mkdir(parents=True, exist_ok=True)
    if encoder == "handcrafted":
        img, txt = handcrafted_features(items, titles, cover)
    else:
        img, txt = hf_features(items, titles, cover, encoder)
    np.savez_compressed(f, items=items, img=img, txt=txt)
    return img, txt
