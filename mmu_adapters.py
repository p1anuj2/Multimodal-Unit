import torch.nn as nn

class ModalityAdapter(nn.Module):
    """
    Projects raw modality features to a shared hidden space and normalizes them.
    Keep it light: a single linear projection + LayerNorm is sufficient and fast.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, T, input_dim]
        return self.norm(self.proj(x))
