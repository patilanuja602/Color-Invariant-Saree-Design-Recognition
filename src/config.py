"""Paths and constants. Values marked [artifact] come from the training notebook / results.json."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.getenv("SAREE_WEIGHTS", ROOT / "weights" / "saree_embedder_slim.pt"))
WEIGHTS_URL = os.getenv("SAREE_WEIGHTS_URL", "https://github.com/patilanuja602/Color-Invariant-Saree-Design-Recognition/releases/download/v1.0.0/saree_embedder_slim.pt")          # direct link (GitHub Release / HF Hub) used if file is missing
GALLERY_DIR = Path(os.getenv("SAREE_GALLERY", ROOT / "gallery"))
EVAL_DIR = ROOT / "evaluation"
FIG_DIR = ROOT / "figures"
STORE_SIZE = 256      # [artifact] stored at 256x256 (bicubic) first
IMG_SIZE = 224        # [artifact] then bilinear+antialias to 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # [artifact] ImageNet
EMB_DIM = 256         # [artifact]
DEFAULT_THRESHOLD = 0.7033937573432922   # [artifact] B3_final, val-only max-F1 threshold
DESIGN_AGG = "max"    # max | topn_mean   (design choice, not trained)
TOP_N = 2
TOP_DESIGNS = 5
AMBIGUITY_RATIO = 0.75
