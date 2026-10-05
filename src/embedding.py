import numpy as np, torch
from .preprocessing import load_rgb, to_tensor

class Embedder:
    def __init__(self, model, device="cpu"): self.model, self.device = model, device
    @torch.no_grad()
    def embed_arrays(self, arr: np.ndarray, bs=32) -> np.ndarray:
        out = [self.model(to_tensor(arr[i:i+bs], self.device)).cpu() for i in range(0, len(arr), bs)]
        return torch.cat(out).numpy().astype("float32")
    def embed(self, src) -> np.ndarray:
        return self.embed_arrays(load_rgb(src)[None])[0]
    def embed_array(self, arr: np.ndarray) -> np.ndarray:
        return self.embed_arrays(arr[None])[0]
