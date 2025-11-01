import torch.nn as nn

class TransformerFusion(nn.Module):
    """
    Single-stream transformer encoder. Receives already-projected tokens from all modalities.
    """
    def __init__(self, hidden_dim=768, num_layers=12, num_heads=12, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, T_total, D]
        return self.encoder(x)
