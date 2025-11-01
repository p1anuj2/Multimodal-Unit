import os
import json
import torch
from torch.utils.data import Dataset

class JsonPairsDataset(Dataset):
    """
    Generic JSONL/JSON dataset for (image_path, text, tokens, label) style entries.
    This aligns with the 'data/processed/*.json' format produced by preprocess scripts.
    If no file is found, the dataset falls back to a synthetic toy dataset (for sanity runs).
    """
    def __init__(self, json_path: str = "data/processed/coco_train.json",
                 vision_dim: int = 2048, text_dim: int = 768, toy_size: int = 256):
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                self.samples = json.load(f)
            self.toy = False
        else:
            # Toy mode: small random tensors to keep the pipeline executable without data.
            self.samples = [{"image_path": "", "text": "", "tokens": [], "label": 1}
                            for _ in range(toy_size)]
            self.toy = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        if self.toy:
            vtoks = torch.randn(16, self.vision_dim)   # 16 visual tokens
            ttoks = torch.randn(32, self.text_dim)     # 32 textual tokens
            label = torch.tensor(1)                    # dummy label
            return vtoks, ttoks, label

        # In a real setup, you would: load image -> encode -> visual tokens,
        # tokenize text -> text tokens. Here we assume preprocess saved token arrays.
        # For simplicity, we simulate with random tokens but keep the interface identical.
        vtoks = torch.randn(16, self.vision_dim)
        ttoks = torch.randn(32, self.text_dim)
        label = torch.tensor(s.get("label", 1))
        return vtoks, ttoks, label
