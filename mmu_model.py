import torch
import torch.nn as nn
from .adapters import ModalityAdapter
from .fusion import TransformerFusion

class MultiModalUNIT(nn.Module):
    """
    Multi-Modal UNIT (MMU)
    - Single-stream encoder over concatenated visual + textual tokens.
    - Lightweight modality adapters isolate modality-specific projection/normalization.
    - A pooled token is used for global outputs (e.g., retrieval alignment, classification).
    """
    def __init__(
        self,
        vision_dim=2048,
        text_dim=768,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        dropout=0.1,
        cls_head_dim=768,
    ):
        super().__init__()
        self.v_adapter = ModalityAdapter(vision_dim, hidden_dim)
        self.t_adapter = ModalityAdapter(text_dim, hidden_dim)
        self.fusion = TransformerFusion(hidden_dim, num_layers, num_heads, dropout)

        # Learned CLS token to aggregate sequence information
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Projection head for global representation
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, cls_head_dim),
        )

    def forward(self, v_tokens, t_tokens):
        """
        Args:
            v_tokens: [B, Tv, vision_dim]
            t_tokens: [B, Tt, text_dim]
        Returns:
            pooled: [B, cls_head_dim] – pooled representation for downstream objectives.
        """
        v = self.v_adapter(v_tokens)  # [B, Tv, D]
        t = self.t_adapter(t_tokens)  # [B, Tt, D]

        x = torch.cat([v, t], dim=1)  # single-stream
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)          # prepend CLS

        z = self.fusion(x)                      # [B, 1 + Tv + Tt, D]
        pooled = z[:, 0]                        # CLS
        return self.head(pooled)                # [B, cls_head_dim]
