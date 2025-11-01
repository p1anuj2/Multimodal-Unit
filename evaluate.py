import argparse
import torch
from mmu import MultiModalUNIT
from training.utils import set_seed
from data.datasets import JsonPairsDataset
from torch.utils.data import DataLoader

def parse_args():
    ap = argparse.ArgumentParser("Evaluate MMU")
    ap.add_argument("--val_json", type=str, default="data/processed/coco_val.json",
                    help="If missing, evaluation runs in toy mode.")
    ap.add_argument("--checkpoint", type=str, default="results/checkpoints/mmu.pth")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset (toy mode if file missing)
    dataset = JsonPairsDataset(json_path=args.val_json)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Build model and load checkpoint
    model = MultiModalUNIT().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Simple embedding norm summary as a sanity “evaluation”
    # (Replace with task-specific metrics when real heads/datasets are wired)
    total_norm = 0.0
    n = 0
    with torch.no_grad():
        for v, t, _ in loader:
            v, t = v.to(device), t.to(device)
            z = model(v, t)
            total_norm += z.norm(dim=-1).sum().item()
            n += z.size(0)

    avg_norm = total_norm / max(1, n)
    print(f"Evaluation summary: avg_embedding_norm={avg_norm:.4f} over {n} samples")
    print("Evaluation complete (demo). For task metrics, integrate respective heads/dataloaders.")

if __name__ == "__main__":
    main()
