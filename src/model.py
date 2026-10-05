"""Re-implementation of SareeEmbedder from the training notebook.
backbone DINOv2-S/14 (HF Dinov2Model); pooled = concat[CLS, mean(patch tokens)] (768-d);
head = LayerNorm -> Dropout -> Linear(768,1024) -> GELU -> Linear(1024,256); output L2-normalised.
cat_head is in the state_dict (aux_category_weight=0 -> unused at inference)."""
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import Dinov2Config, Dinov2Model

def dinov2_small() -> Dinov2Model:   # facebook/dinov2-small architecture; weights come from our checkpoint
    return Dinov2Model(Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                                    patch_size=14, image_size=518, mlp_ratio=4))

class SareeEmbedder(nn.Module):
    def __init__(self, backbone, emb_dim=256, hidden=1024, dropout=0.1, n_cat=4):
        super().__init__()
        self.backbone = backbone
        d = backbone.config.hidden_size * 2
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, emb_dim))
        self.cat_head = nn.Linear(emb_dim, n_cat)

    def forward(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
        return F.normalize(self.head(f).float(), dim=-1)

def load_model(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = SareeEmbedder(dinov2_small(), c["emb_dim"], c["head_hidden"], c["head_dropout"])
    sd = ck["state_dict"]
    nsd = {}
    for k, v in sd.items():
        nk = k
        nk = nk.replace('.attention.attention.query.', '.attention.q_proj.')
        nk = nk.replace('.attention.query.', '.attention.q_proj.')
        nk = nk.replace('.attention.attention.key.', '.attention.k_proj.')
        nk = nk.replace('.attention.key.', '.attention.k_proj.')
        nk = nk.replace('.attention.attention.value.', '.attention.v_proj.')
        nk = nk.replace('.attention.value.', '.attention.v_proj.')
        nk = nk.replace('.attention.output.dense.', '.attention.o_proj.')
        nsd[nk] = v
    m.load_state_dict(nsd, strict=True)
    return m.to(device).eval(), dict(config=c, threshold=float(ck.get("threshold", 0)), label_source=ck.get("label_source"))
