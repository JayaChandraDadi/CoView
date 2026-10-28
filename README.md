# CoView

Multimodal item embeddings for cold-start retrieval. A frozen vision-language encoder (CLIP or SigLIP) reads each short video's cover and title; a small fusion head is trained with symmetric InfoNCE on **co-engagement pairs** (items the same user engaged with close together), so new items land next to the items people actually watch alongside them, not just the ones that look similar.

This is the approach behind recent industry work on multimodal item-to-item retrieval: content backbone for coverage, behavioural supervision for relevance.

```
cover ──► CLIP image encoder (frozen) ──► MLP ─┐
                                                ├─► learned gate ─► 128-d item vector ─► FAISS / ReelFunnel
title ──► CLIP text encoder (frozen)  ──► MLP ─┘
            trained with InfoNCE on (item_a, item_b) co-engagement pairs, in-batch negatives
```

## What is in it

- **Co-engagement pairs** mined from user sequences (items within 5 consecutive interactions), canonicalised and deduplicated.
- **Fusion head**: one MLP per modality, a learned softmax gate, L2-normalised output, CLIP-style learnable temperature.
- **Modality dropout** (30% of rows lose image or text, never both) so one checkpoint embeds an item from its cover alone, its title alone, or both. Cold-start traffic really does arrive with missing titles and broken covers.
- **Cold-start evaluation**: a time split where "cold" items have zero training interactions. Each user's training history is the query; only cold items are candidates. An ID-embedding model cannot place these items at all; it scores at chance.
- **Export**: `item_embeddings.npz` (item id + vector), which ReelFunnel's item tower reads with `--content-emb`.

Encoder features are computed once and cached, so training touches only the head and runs in minutes on an M-series Mac.

## Results (synthetic MicroLens-layout data: 119K interactions, 2,541 warm and 448 cold items)

Retrieving cold items for 2,785 test users:

| Method | Recall@10 | Recall@50 | NDCG@10 |
| --- | --- | --- | --- |
| Chance (an ID embedding with no history) | 0.022 | 0.112 | |
| Frozen encoder, zero-shot | 0.239 | 0.507 | 0.151 |
| CoView, image only | 0.115 | 0.370 | 0.067 |
| CoView, text only | 0.295 | 0.734 | 0.175 |
| **CoView, fused** | **0.287** | **0.755** | **0.173** |

Co-engagement training lifts cold-start Recall@50 by 49% over the same frozen features used zero-shot, 6.8x over chance. Head training: 168 s on 2 CPU cores, 289K pairs.

These numbers use the `handcrafted` encoder (colour histograms and hashed bag-of-words), the offline fallback for machines without the CLIP download. They prove the pipeline, not CLIP's quality. Run with `--encoder clip` on real MicroLens for numbers you can report.

## Run it

```bash
pip install -r requirements.txt

# Synthetic (covers are drawn images, titles are topic words)
python -m coview.cli synth --out data/synth
python -m coview.cli train --data data/synth --encoder handcrafted --out artifacts

# Real: MicroLens (https://github.com/westlake-repl/MicroLens). Point at its interaction file
# (user, item, timestamp), its English title file (item, title) and its cover folder:
python -m coview.cli train --data data/microlens --pairs <interactions.csv> \
    --titles <titles.csv> --covers <covers_dir> --encoder clip

pytest -q
```

`--encoder siglip` swaps in SigLIP; any Hugging Face model id with `get_image_features` / `get_text_features` works. The first CLIP run downloads about 600 MB and encodes on MPS when available.

## Layout

```
coview/data.py      MicroLens loader, cold-start split, co-engagement pair mining, synthetic generator
coview/encoders.py  CLIP/SigLIP feature extraction (cached) and the offline fallback
coview/model.py     fusion head, modality masks, symmetric InfoNCE, training loop
coview/cli.py       train → evaluate → export
tests/              split, pairs, modality modes, loss, synthetic layout
```

## Design notes

- **Why behaviour, not just similarity.** Zero-shot CLIP puts visually similar covers together. Co-engagement supervision teaches the head which content signals predict what people watch next. That is the jump from 0.51 to 0.75 above.
- **Why freeze the backbone.** On a laptop, cached features make every epoch seconds instead of hours, and the head is where the behavioural signal goes. Unfreezing the last block is the natural next experiment when there is a GPU.
- **Text dominates here** because the synthetic titles carry topic words more reliably than the drawn covers. Real data will weigh the modalities differently; the per-mode rows show which one carries your dataset.
