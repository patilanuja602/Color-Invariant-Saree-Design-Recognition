import numpy as np, pandas as pd
from pathlib import Path
from . import config as C

class Gallery:
    def __init__(self, emb, meta, root=C.GALLERY_DIR):
        assert len(emb) == len(meta), "embeddings/metadata length mismatch"
        self.emb, self.meta, self.root = emb.astype("float32"), meta.reset_index(drop=True), Path(root)
        if "category" not in self.meta: self.meta["category"] = np.nan
    @classmethod
    def load(cls, root=C.GALLERY_DIR):
        root = Path(root)
        return cls(np.load(root / "embeddings.npy"), pd.read_csv(root / "metadata.csv"), root)
    def image_path(self, i): return self.root / self.meta.at[i, "image_path"]
    @property
    def n_designs(self): return self.meta.design_id.nunique()
