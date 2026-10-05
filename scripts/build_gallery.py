"""Build gallery/{images,metadata.csv,embeddings.npy}.
Mode A (--splits-csv): public Kaggle gallery from evaluation/splits.csv (source=kaggle, --splits), images found under --kaggle-dir.
Mode B (--metadata): your CSV with image_path[,image_id,design_id,colorway_id,category,source]; paths relative to --images-root.
Missing category stays empty -> shown as 'Unknown / unavailable'. Nothing is invented."""
import argparse, shutil, sys, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.model import load_model
from src.embedding import Embedder
from src.preprocessing import load_rgb, InvalidImage

p = argparse.ArgumentParser()
p.add_argument("--weights", default=str(C.WEIGHTS)); p.add_argument("--out", default=str(C.GALLERY_DIR))
p.add_argument("--splits-csv"); p.add_argument("--kaggle-dir"); p.add_argument("--splits", default="train,val")
p.add_argument("--metadata"); p.add_argument("--images-root", default=".")
a = p.parse_args(); out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)

if a.splits_csv:
    s = pd.read_csv(a.splits_csv); s = s[(s.source == "kaggle") & s.split.isin(a.splits.split(",")) & s.usable]
    idx = {f.name: f for f in Path(a.kaggle_dir).rglob("*") if f.is_file()}
    s = s[s.name.isin(idx)]; assert len(s), "no Kaggle files matched splits.csv names"
    rows = []
    for i, r in enumerate(s.itertuples()):
        dst = f"images/k{i:04d}{Path(r.name).suffix}"; shutil.copy(idx[r.name], out / dst)
        rows.append(dict(image_id=f"k{i:04d}", image_path=dst, design_id=r.design_id, category=r.category, source="kaggle", split=r.split))
    meta = pd.DataFrame(rows)
else:
    meta = pd.read_csv(a.metadata); root = Path(a.images_root)
    if "image_id" not in meta: meta.insert(0, "image_id", [f"g{i:04d}" for i in range(len(meta))])
    if "design_id" not in meta: meta["design_id"] = meta.image_id   # no design ids -> each image is its own design
    new = []
    for r in meta.itertuples():
        dst = f"images/{r.image_id}{Path(r.image_path).suffix}"; shutil.copy(root / r.image_path, out / dst); new.append(dst)
    meta["image_path"] = new

dev = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = load_model(a.weights, dev); emb = Embedder(model, dev)
keep, arrs = [], []
for i, r in enumerate(meta.itertuples()):
    try: arrs.append(load_rgb(out / r.image_path)); keep.append(i)
    except InvalidImage as e: print("skip", r.image_path, e)
meta = meta.iloc[keep].reset_index(drop=True)
Z = emb.embed_arrays(np.stack(arrs)); assert np.allclose(np.linalg.norm(Z, axis=1), 1, atol=1e-3)
np.save(out / "embeddings.npy", Z); meta.to_csv(out / "metadata.csv", index=False)
print("gallery", Z.shape, "designs", meta.design_id.nunique(), meta["category"].value_counts().to_dict() if "category" in meta else "")
