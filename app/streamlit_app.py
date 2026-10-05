import sys, json, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, streamlit as st, torch
from PIL import Image
from src import config as C
from src.model import load_model
from src.embedding import Embedder
from src.gallery import Gallery
from src.preprocessing import load_rgb, InvalidImage
from src.retrieval import design_search, image_search
from src.category import aggregate

st.set_page_config(page_title="Saree Design Identification", page_icon="🧵", layout="wide")

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    if not C.WEIGHTS.exists():
        if not C.WEIGHTS_URL: return None, None
        C.WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(C.WEIGHTS_URL, C.WEIGHTS)
        except Exception:
            return None, None
    try:
        m, info = load_model(C.WEIGHTS, "cpu")
        return Embedder(m, "cpu"), info
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        return None, None

@st.cache_resource(show_spinner="Loading gallery…")
def get_gallery():
    try: return Gallery.load()
    except Exception: return None

@st.cache_data
def get_results():
    p = C.EVAL_DIR / "results.json"
    if not p.exists(): return None
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return None

E, INFO = get_model(); G = get_gallery()
TAU = (INFO or {}).get("threshold") or C.DEFAULT_THRESHOLD
page = st.sidebar.radio("Page", ["Identify Saree", "Gallery", "Evaluation"])
if E is None: st.error("Model weights not found. See README → 'Weights'."); st.stop()
def need_gallery():
    if G is None: st.warning("No gallery found. Run scripts/build_gallery.py (see README)."); st.stop()
def read(upl):
    try: return load_rgb(upl)
    except InvalidImage as e: st.error(str(e)); return None
def gimg(i): return Image.open(G.image_path(i))

if page == "Identify Saree":
    need_gallery(); st.title("Saree Design Identification")
    up = st.file_uploader("Upload an RGB saree image", type=["jpg", "jpeg", "png", "webp"])
    if up:
        st.image(up, caption="Query image", width=280)
        if st.button("Find Matching Designs", type="primary"):
            arr = read(up)
            if arr is not None:
                q = E.embed_array(arr); D = design_search(q, G); R = aggregate(D, TAU)
                if R["status"] == "unknown":
                    st.warning(f"**NO SUFFICIENTLY SIMILAR KNOWN DESIGN FOUND**  \nBest design similarity {R['best']:.3f} < threshold {TAU:.3f}.")
                else:
                    st.success(f"### IDENTIFIED SAREE TYPE\n## ★ {R['category'].upper()}" if R["category_known"] else "### Category unavailable for matched designs (gallery has no saree-type metadata for them)")
                    st.metric("Best design match (cosine similarity)", f"{R['top_score']*100:.1f}%")
                    st.caption(f"Supporting: {R['n_supporting']} retrieved design(s) in this category above the threshold. Match strength = cosine similarity, not a probability.")
                    if R["ambiguous"]: st.info(f"MOST LIKELY: **{R['category']}** — ALTERNATIVE: **{', '.join(R['alternatives'])}** (evidence is close)")
                st.subheader("Top matching reference designs (one row per design)")
                cols = st.columns(min(5, len(D)))
                for c, (_, r) in zip(cols, D.head(5).iterrows()):
                    c.image(gimg(r.best_idx), use_container_width=True)
                    c.markdown(f"**{r.design_id}**  \nSimilarity {r.design_score*100:.1f}%  \nType: {r.category if isinstance(r.category, str) else 'Unknown / unavailable'}  \n{r.n_images} gallery image(s)")
                if R["status"] == "ok":
                    st.subheader("Why this result?")
                    for c, v, share, n in R["evidence"]:
                        st.write(f"{c} — {n} design(s)"); st.progress(float(share))
                    st.caption("Bars show each category's share of similarity-weighted retrieval evidence (sum of sim − threshold over top-5 designs). Not a calibrated probability.")
                with st.expander("Image-level matches"):
                    st.dataframe(image_search(q, G, 10)[["similarity", "design_id", "category", "image_id"]])

elif page == "Gallery":
    need_gallery(); st.title("Reference Gallery"); m = G.meta
    st.write(f"{len(m)} images · {G.n_designs} designs")
    f = st.columns(2)
    cat = f[0].multiselect("Saree type", sorted(m.category.dropna().unique())) if m.category.notna().any() else []
    dz = f[1].text_input("Design ID contains")
    v = m
    if cat: v = v[v.category.isin(cat)]
    if dz: v = v[v.design_id.astype(str).str.contains(dz)]
    cols = st.columns(5)
    for k, (i, r) in enumerate(v.head(60).iterrows()):
        c = cols[k % 5]; c.image(gimg(i), use_container_width=True)
        c.caption(" · ".join([f"{r.design_id}", f"type: {r.category}" if isinstance(r.category, str) else "type: Unknown / unavailable"] + ([f"colorway: {r.colorway_id}"] if "colorway_id" in r and pd.notna(r.colorway_id) else [])))

elif page == "Evaluation":
    st.title("Evaluation (from results_bundle / figures_bundle)"); R = get_results()
    if R is None:
        st.error("Evaluation results file (results.json) not found or invalid.")
        st.stop()
    fm = R["final_model"]
    st.warning(f"Labels: **{R['label_source']}** — DeepLure design/colorway ids were auto-proposed (no reliable ids in the raw corpus); results are against pseudo-labels. Query sets are tiny (val 20, test 8 design queries) → wide CIs.")
    st.write(f"Final model: **{fm}** · verification threshold (validation-derived): **{R['final_threshold']:.4f}**")
    def tbl(split, name):
        r = R["results"][name][split]; d = r["design"]; v = r["verif_all"]; vs = r["verif_neg_samecolor"]
        return {"Recall@1": d["R@1"], "Recall@5": d["R@5"], "Recall@10": d["R@10"], "mAP": d["mAP"], "MRR": d["MRR"], "Cross-colour R@1": r["cross"]["R@1"],
                "Accuracy": v["accuracy"], "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1"], "ROC-AUC": v["roc_auc"], "PR-AUC": v["pr_auc"], "EER": v["eer"],
                "Same-colour-negatives ROC-AUC": vs["roc_auc"]}
    for split in ("val", "test"):
        st.subheader(f"{split.upper()} RESULTS" + (" (used for threshold/model decisions)" if split == "val" else " (held out)"))
        st.dataframe(pd.DataFrame({n: tbl(split, n) for n in R["results"]}).round(3))
        d = R["results"][fm][split]; st.caption(f"{fm}/{split}: queries {d['design']['n_queries']} · gallery {d['design']['gallery']} · designs {d['design']['designs']} · colourways {d['design']['colorways']} · verification pairs {d['verif_all']['n']} ({d['verif_all']['pos']} positive) · TP/FP/TN/FN {d['verif_all']['TP']}/{d['verif_all']['FP']}/{d['verif_all']['TN']}/{d['verif_all']['FN']}")
    st.subheader("TRAINING")
    st.write("Only the training loss / validation-mAP history is stored (no separate training-set retrieval metrics were computed).")
    hist_file = C.EVAL_DIR / "history_B3_final.csv"
    if hist_file.exists():
        st.dataframe(pd.read_csv(hist_file).round(4))
    else:
        st.info("Training history file not found.")
    for f in ("val_curve.png", "pair_similarity.png"):
        if (C.FIG_DIR / f).exists(): st.image(str(C.FIG_DIR / f), caption=f)
    ce = C.EVAL_DIR / "category_eval.json"
    st.subheader("Saree-type readout (Kaggle test split)")
    st.json(json.loads(ce.read_text())) if ce.exists() else st.info("Not yet computed — run scripts/evaluate.py.")

