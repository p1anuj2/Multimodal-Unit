import argparse
import torch
from torch.utils.data import DataLoader
from mmu import MultiModalUNIT, contrastive_loss
from training.utils import set_seed, move_to_device, save_checkpoint
from data.datasets import JsonPairsDataset

def parse_args():
    ap = argparse.ArgumentParser("Train MMU")
    ap.add_argument("--train_json", type=str, default="data/processed/coco_train.json",
                    help="Path to processed training JSON. Falls back to toy mode if missing.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--save", type=str, default="results/checkpoints/mmu.pth")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset & Loader
    dataset = JsonPairsDataset(json_path=args.train_json)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Model
    model = MultiModalUNIT(
        vision_dim=2048,
        text_dim=768,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training on device={device} | epochs={args.epochs} | batches per epoch={len(loader)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            v, t, _ = move_to_device(batch, device)  # labels unused in this simple alignment loop

            # Two views: in a more advanced setup, you would encode images/text separately.
            zi = model(v, t)   # global pooled rep
            zt = model(v, t)   # second branch (placeholder for text pathway head)
            loss = contrastive_loss(zi, zt)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += loss.item()

        avg = running / max(1, len(loader))
        print(f"[Epoch {epoch:02d}] train_loss={avg:.4f}")

    # Save a minimal checkpoint for evaluation/demo.
    save_checkpoint({"model_state": model.state_dict(), "args": vars(args)}, args.save)
    print(f"Checkpoint saved to {args.save}")

if __name__ == "__main__":
    main()
