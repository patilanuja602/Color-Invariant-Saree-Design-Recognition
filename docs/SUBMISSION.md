# Submission pack

## Approach note (≤500 chars)
DINOv2-S/14 (last 4 blocks tuned) + MLP head → 256-d L2 embedding; cosine retrieval, threshold chosen on val. Input: 256²→224², ImageNet norm. Two-view SupCon (T=0.1), 8 designs×4 imgs; GPU colour aug (hue, channel-permute, gray, gamma) removes colour shortcuts; same-colour hard negatives + pHash false-negative guard; dedup'd Kaggle as extra negatives. Design-disjoint splits; R@k/mAP/MRR, ROC/PR-AUC, EER. Saree type = similarity-weighted vote over retrieved designs.

## Interview defence (short answers)
1. **Retrieval, not classification** – designs are open-set; gallery grows without retraining; type is read from retrieved metadata.
2. **Colour invariance** – colour augmentation in training (two views of one image with different colour transforms are positives) + same-colour hard negatives.
3. **Colour shortcut** – channel permutation/hue/gray aug; frozen-gray baseline (B4) as control; same-colour-negative ROC-AUC reported (test 0.896).
4/5. Same motif/different colour → positives across colourways + recolour views; different motif/same colour → hard-negative batches.
6. **DINOv2-S** – strong self-supervised texture/structure features, small (23 M), 10 ms GPU.
7. **256-d** – compact, 1 dot product/gallery item; not ablated (say so).
8. **Cosine** – L2-normalised embeddings, matches SupCon objective.
9. Positives: auto-proposed colourway clusters + two augmented views. 10. Hard negatives: same-colour different-design in batch, guarded by pHash/similarity quantile (0.995) mask.
11. **Leakage** – split_group-based design-disjoint splits; Kaggle pHash dedup (thr 6); original Kaggle split not trusted.
12. Gallery/query: per-design colourway held out as query vs rest as gallery (see notebook `evaluate_embeddings`) — confirm the exact rule in the notebook before the call.
13. **Threshold** – max-F1 on validation pairs (0.703); test never used.
14/15. Weighted evidence over top-5 retrieved designs; no classifier head used (aux weight 0).
16. Best design < τ → "No sufficiently similar known design found".
17. Pseudo-labels, tiny test (8 queries), low verification precision, test R@1 0.50 < frozen baseline 0.75, Kaggle category ≠ design.
18. New garments: new gallery; embeddings are texture-generic; re-fine-tune if domain gap.

## Final checklist
- [ ] Run `pytest -q`, build gallery, run app locally, run `evaluate.py`
- [ ] Check gallery image licence; no DeepLure images/embeddings/filenames in repo (use slim checkpoint; don't commit `figures/grid_*.png` — they show DeepLure crops)
- [ ] Weights as Release asset + `SAREE_WEIGHTS_URL` secret; deploy; open the URL in a private window
- [ ] Repo public, notebook viewable, form submitted before 11:59 PM IST 5 Oct 2026
- [ ] Delete DeepLure copy after the exercise (JD requirement)
