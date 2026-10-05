# Saree Design Recognition — colour-invariant design retrieval ("face recognition for textiles")

## 1–3. Overview, problem, objective
Identify a saree by the **design on its surface**, independent of colour palette. Same design + different colour ⇒ match; different design + same colour ⇒ no match.
Core ML task = **design retrieval** (embedding + gallery search) and **pair verification**. *Saree type* is a user-facing readout of the retrieved reference designs' metadata — **not** a classifier.

## 4–5. Architecture & model (audited from the training notebook / checkpoint)
```
image → EXIF/RGB → 256² bicubic → 224² bilinear → ImageNet norm → DINOv2-S/14 → [CLS ‖ mean patch] (768)
      → LayerNorm → Dropout → Linear(768,1024) → GELU → Linear(1024,256) → L2-norm → 256-d embedding
embedding ─┬─ cosine vs cached gallery → design-level aggregation → category evidence → saree type / "unknown"
           └─ cosine vs second image  → SAME / DIFFERENT (validation-derived threshold 0.7034)
```
23.1 M params (8.15 M tuned in final stage), 12.25 GFLOPs/img, 256-d embedding (numbers from `evaluation/results.json`).

## 6–7. Training & colour invariance (from notebook)
Frozen warm-up (8 ep) then last 4 DINOv2 blocks (LR 1e-5) + head (3e-4), AdamW wd 0.05, warm-up + schedule, early stopping on val mAP. Two-view **supervised contrastive** loss (T=0.1), 8 designs × 4 images per batch. GPU colour augmentation (hue, channel permutation, saturation, gamma, grayscale) so colour cannot be a shortcut; batches include **same-colour / different-design hard negatives** with a pHash/similarity **false-negative guard**; de-duplicated Kaggle images serve as extra negatives. Ablations stored: B1 frozen RGB, B4 frozen gray, B2 fine-tuned+colour-aug, B3 final.

## 8–10. Retrieval, verification, saree type
* **Retrieval**: cosine on L2-normalised vectors (= dot product). Images are collapsed to **designs** (`max` over a design's colourways by default; `topn_mean` configurable) so one design is never listed several times.
* **Verification**: cosine ≥ τ, τ = 0.7034 = max-F1 on **validation** pairs (never tuned on test).
* **Saree type**: `score(c) = Σ_{top-5 designs in c} max(0, sim − τ)`. Similarity-weighted, each design counted once, only designs above τ count. If the best design < τ → *"No sufficiently similar known design found."* If the runner-up ≥ 75 % of the winner → shown as *Alternative*. Scores are retrieval evidence, **not probabilities**. Motif names are never shown (none exist in metadata).
* **Extensibility**: add labelled images of new types (Kanjeevaram, Paithani…) to the gallery and re-run `build_gallery.py` — no retraining.

## 11–12. Evaluation & results
Full tables are in the app (Evaluation page) and `evaluation/results_{val,test}.csv`. Final model **B3_final**, design-disjoint split, DeepLure test = 8 queries / 28 gallery images / 23 designs; val = 20 / 31 / 19.

| split | R@1 | R@5 | R@10 | mAP | MRR | ver. ROC-AUC | PR-AUC | EER | F1 | precision | recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 0.90 | 1.00 | 1.00 | 0.917 | 0.95 | 0.937 | 0.447 | 0.108 | 0.643 | 0.47 | 1.00 |
| test | 0.50 | 1.00 | 1.00 | 0.764 | 0.75 | 0.924 | 0.375 | 0.171 | 0.370 | 0.23 | 1.00 |

**Honest reading**: R@5/R@10 are saturated because galleries are tiny; test R@1 rests on 8 queries (bootstrap CI in CSV is very wide); at τ=0.703 recall is 1.0 but precision is low (many false positives, esp. same-colour negatives) and the frozen DINOv2 baselines scored higher test R@1 (0.75). We do **not** swap models after seeing test. Saree-type accuracy is **not in the bundle** → `scripts/evaluate.py` computes it (output `evaluation/category_eval.json`).

## 13–15. Data, limitations, pretrained resources
* **DeepLure corpus** (proprietary, not redistributed): no reliable design ids → **auto-proposed pseudo-labels** (`label_source=auto_proposed`), 5 images excluded as blank, some low-texture. **No saree-type metadata.**
* **Kaggle Indian Saree Patterns**: category labels only (Banarasi 132 / Pichwai 94 / Bandhani 92 / Ikat 92 in `splits.csv`); heavy augmentation duplicates → de-duplicated by pHash, each kept image gets a unique pseudo `design_id` (`kaggle_N`) — **category ≠ design id**.
* Therefore the saree-type gallery is the Kaggle gallery; DeepLure images can be added by you privately (type will show *Unknown / unavailable*).
* Pretrained: `facebook/dinov2-small` (Meta). External data: Kaggle dataset (disclosed).

## 16–19. Run it
```bash
pip install -r requirements.txt
# 1. slim checkpoint (removes DeepLure gallery embeddings/filenames baked into the training .pt)
python scripts/export_slim.py /path/saree_design_embedder.pt weights/saree_embedder_slim.pt
# 2. public gallery from the Kaggle download (unzipped anywhere) using the documented split
python scripts/build_gallery.py --splits-csv evaluation/splits.csv --kaggle-dir /path/to/kaggle --splits train,val
python scripts/evaluate.py --kaggle-dir /path/to/kaggle        # saree-type accuracy on Kaggle test split
streamlit run app/streamlit_app.py
pytest -q
```
Private gallery: `python scripts/build_gallery.py --metadata my.csv --images-root /private/imgs --out gallery_private` then `SAREE_GALLERY=gallery_private streamlit run …`. CSV columns: `image_path` (+ optional `design_id, colorway_id, category`).

## Deployment (Streamlit Community Cloud)
Slim weights ≈ 90 MB: keep out of git (`.gitignore`); upload as a **GitHub Release asset** and set secret `SAREE_WEIGHTS_URL = "<direct asset URL>"` (the app downloads once, cached). Commit `gallery/` (Kaggle images + `metadata.csv` + `embeddings.npy`) only after confirming the dataset licence allows redistribution; otherwise host the gallery the same way. CPU-only torch is used; expect ~1 GB RAM. Main file: `app/streamlit_app.py`.

## 20. Structure
`app/` UI · `src/` engine (UI-independent) · `scripts/` build/evaluate/export · `tests/` · `evaluation/` shipped results · `figures/` · `notebooks/training.ipynb` · `gallery/` · `docs/`.

## 21–23. Reproducibility, limitations, future
Seed 42, config in `evaluation/results.json`. Limitations: pseudo-labels, tiny splits, low verification precision, Kaggle categories are image-level not design-level, no motif names. Future: other garments = new gallery + (optionally) re-fine-tune with garment-specific design ids; manual verified design CSV (`use_verified_csv`) is the highest-value upgrade.

> Not executed here: this build environment had no PyTorch/network, so the app, tests and model loading are **written but unrun**. Run `pytest -q` first; see docs/SUBMISSION.md.
