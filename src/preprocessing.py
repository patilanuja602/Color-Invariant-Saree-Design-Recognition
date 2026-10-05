"""Matches notebook: EXIF-transpose -> RGB -> 256x256 bicubic -> /255 -> 224 bilinear(antialias) -> ImageNet norm."""
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from . import config as C

class InvalidImage(ValueError): pass

def load_rgb(src, size=C.STORE_SIZE) -> np.ndarray:
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if min(im.size) < 16: raise InvalidImage("Image too small")
            return np.asarray(im.resize((size, size), Image.BICUBIC), dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImage(f"Not a readable image: {e}")

def to_tensor(batch: np.ndarray, device="cpu") -> torch.Tensor:
    x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div(255).to(device)
    x = F.interpolate(x, size=(C.IMG_SIZE, C.IMG_SIZE), mode="bilinear", antialias=True, align_corners=False)
    m = torch.tensor(C.MEAN, device=device).view(1, 3, 1, 1); s = torch.tensor(C.STD, device=device).view(1, 3, 1, 1)
    return (x - m) / s
