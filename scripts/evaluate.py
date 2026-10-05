"""NEW evaluation (absent from results_bundle): saree-type readout accuracy.
Gallery = Kaggle train+val (build_gallery --splits-csv ...). Queries = Kaggle TEST rows of evaluation/splits.csv
(split by the notebook's near-duplicate-aware groups). Threshold = validation-derived, not tuned here."""
import argparse, sys, json, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.model import load_model; from src.embedding import Embedder; from src.gallery import Gallery
from src.retrieval import design_search; from src.category import aggregate
p = argparse.ArgumentParser(); p.add_argument("--kaggle-dir", required=True); p.add_argument("--weights", default=str(C.WEIGHTS))
a = p.parse_args()
G = Gallery.load(); model, info = load_model(a.weights); E = Embedder(model)
s = pd.read_csv(C.EVAL_DIR / "splits.csv"); s = s[(s.source == "kaggle") & (s.split == "test") & s.usable]
idx = {f.name: f for f in Path(a.kaggle_dir).rglob("*") if f.is_file()}; s = s[s.name.isin(idx)]
tau = info["threshold"] or C.DEFAULT_THRESHOLD; res = []
for r in s.itertuples():
    o = aggregate(design_search(E.embed(idx[r.name]), G), tau)
    res.append(dict(true=r.category, pred=o.get("category", "UNKNOWN"), status=o["status"]))
df = pd.DataFrame(res); ok = df[df.status == "ok"]
out = dict(n_queries=len(df), rejected_as_unknown=int((df.status != "ok").sum()), top1_acc_all=float((df.true == df.pred).mean()),
           top1_acc_accepted=float((ok.true == ok.pred).mean()) if len(ok) else None, threshold=tau,
           confusion=pd.crosstab(df.true, df.pred).to_dict())
(C.EVAL_DIR / "category_eval.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
