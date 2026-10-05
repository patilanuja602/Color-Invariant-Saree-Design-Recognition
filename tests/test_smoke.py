import sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pytest, torch
from PIL import Image
from src import config as C
from src.preprocessing import load_rgb, to_tensor, InvalidImage
from src.retrieval import design_search, image_search
from src.category import aggregate
from src.verification import verify
from src.gallery import Gallery

def _img(c=(200, 30, 30)):
    b = io.BytesIO(); Image.fromarray(np.full((300, 200, 3), c, np.uint8)).save(b, "PNG"); b.seek(0); return b

def _gal():
    e = np.eye(4, 8, dtype="float32")
    m = pd.DataFrame(dict(image_id=list("abcd"), image_path=list("abcd"), design_id=["D1", "D1", "D2", "D3"], category=["Banarasi", "Banarasi", "Ikat", None]))
    return Gallery(e, m)

def test_preprocessing():
    a = load_rgb(_img()); assert a.shape == (256, 256, 3) and a.dtype == np.uint8
    assert to_tensor(a[None]).shape == (1, 3, 224, 224)
def test_invalid_image():
    with pytest.raises(InvalidImage): load_rgb(io.BytesIO(b"not an image"))
def test_retrieval_and_design_aggregation():
    G = _gal(); q = np.array([1, .5, 0, 0, 0, 0, 0, 0], "float32"); q /= np.linalg.norm(q)
    assert image_search(q, G).similarity.iloc[0] >= image_search(q, G).similarity.iloc[1]
    D = design_search(q, G); assert D.design_id.is_unique and D.design_id.iloc[0] == "D1" and D.n_images.iloc[0] == 2
def test_category_and_unknown():
    G = _gal(); q = np.eye(8, dtype="float32")[0]; D = design_search(q, G)
    assert aggregate(D, 0.5)["category"] == "Banarasi"
    assert aggregate(D, 1.01)["status"] == "unknown"
def test_verification():
    v = np.eye(8, dtype="float32"); assert verify(v[0], v[0], 0.7)["same"] and not verify(v[0], v[1], 0.7)["same"]
def test_gallery_load():
    if (C.GALLERY_DIR / "embeddings.npy").exists():
        G = Gallery.load(); assert np.allclose(np.linalg.norm(G.emb, axis=1), 1, atol=1e-3)
@pytest.mark.skipif(not C.WEIGHTS.exists(), reason="weights not present")
def test_model_cpu_embedding():
    from src.model import load_model; from src.embedding import Embedder
    m, _ = load_model(C.WEIGHTS, "cpu"); z = Embedder(m).embed(_img())
    assert z.shape == (C.EMB_DIM,) and abs(np.linalg.norm(z) - 1) < 1e-4
