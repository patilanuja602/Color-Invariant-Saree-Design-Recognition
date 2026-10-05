"""Saree type = interpretation of retrieved DESIGN evidence (no classifier head).
score(c) = sum over top-K retrieved designs d of category c of max(0, sim_d - tau), tau = validation threshold.
Similarity-weighted; only designs above the validation-derived recognition threshold count; each design counted once.
Scores are retrieval EVIDENCE, not calibrated probabilities."""
from collections import defaultdict
from . import config as C

def aggregate(designs, tau, k=C.TOP_DESIGNS, ambiguity=C.AMBIGUITY_RATIO):
    top = designs.head(k)
    if len(top) == 0 or top.design_score.iloc[0] < tau:
        return dict(status="unknown", message="No sufficiently similar known design found.",
                    best=float(top.design_score.iloc[0]) if len(top) else None)
    sc, n = defaultdict(float), defaultdict(int)
    for _, r in top.iterrows():
        if r.design_score < tau: continue
        c = r.category if isinstance(r.category, str) else "Unknown / unavailable"
        sc[c] += r.design_score - tau; n[c] += 1
    ranked = sorted(sc.items(), key=lambda x: -x[1]); tot = sum(sc.values()) or 1.0
    win = ranked[0]; alt = [c for c, v in ranked[1:] if v >= ambiguity * win[1]]
    return dict(status="ok", category=win[0], evidence=[(c, v, v / tot, n[c]) for c, v in ranked],
                n_supporting=n[win[0]], alternatives=alt, ambiguous=bool(alt),
                top_design=top.design_id.iloc[0], top_score=float(top.design_score.iloc[0]),
                category_known=win[0] != "Unknown / unavailable")
