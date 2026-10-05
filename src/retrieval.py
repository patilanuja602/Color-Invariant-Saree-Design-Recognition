import numpy as np, pandas as pd
from . import config as C

def image_search(q, G, top_k=10):
    s = G.emb @ q                                  # cosine (both L2-normalised)
    o = np.argsort(-s)[:top_k]
    d = G.meta.iloc[o].copy(); d.insert(0, "gallery_idx", o); d.insert(1, "similarity", s[o]); return d.reset_index(drop=True)

def design_search(q, G, agg=C.DESIGN_AGG, top_n=C.TOP_N, top_k=10):
    """Collapse images -> designs. 'max' (default): a query needs to match ONE colorway of a design, and top-n-mean
    would penalise designs with few gallery images. 'topn_mean': more robust to a single outlier image."""
    s = G.emb @ q; df = G.meta.assign(similarity=s, gallery_idx=np.arange(len(s)))
    rows = []
    for did, g in df.groupby("design_id", sort=False):
        g = g.sort_values("similarity", ascending=False)
        score = g.similarity.iloc[0] if agg == "max" else g.similarity.iloc[:top_n].mean()
        cats = g.category.dropna().unique()
        rows.append(dict(design_id=did, design_score=float(score), n_images=len(g), best_idx=int(g.gallery_idx.iloc[0]),
                         category=cats[0] if len(cats) else None, evidence_idx=g.gallery_idx.iloc[:3].tolist()))
    return pd.DataFrame(rows).sort_values("design_score", ascending=False).head(top_k).reset_index(drop=True)
