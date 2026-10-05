"""Strip the DeepLure-derived gallery (embeddings + filenames) from the training checkpoint -> public-safe file.
usage: python scripts/export_slim.py saree_design_embedder.pt weights/saree_embedder_slim.pt"""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
slim = {k: ck[k] for k in ("config", "backbone", "state_dict", "threshold", "label_source", "approach_note")}
torch.save(slim, dst); print("saved", dst, list(slim))
