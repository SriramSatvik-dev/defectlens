"""
src/training/train.py

Main training loop for DefectLens. Reused for baseline AND every
ablation -- differences are config flags, not separate scripts.

Freeze logic lives in freeze_utils.py, LR scheduling in scheduler.py --
this file focuses purely on the training loop itself: forward, loss,
backward, optimizer step, per-epoch val AUC-ROC, checkpoint on best val AUC.

Usage (from repo root, inside Colab):
    python -m src.training.train --category screw --loss mse --epochs 50
"""

import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

from src.data.dataset import get_datasets
from src.models.encoder import ResNet18Encoder, EfficientNetB0Encoder
from src.models.autoencoder import DefectLensAE
from src.losses.mse_loss import MSELoss
from src.losses.ssim_composite_loss import CompositeLoss
from src.eval.metrics import compute_pixel_error_map, aggregate_score, compute_auc_roc
from src.training.freeze_utils import apply_freeze_config, count_trainable_params
from src.training.scheduler import build_scheduler, step_scheduler


def build_model(args, device):
    if args.encoder_backbone == "resnet18":
        encoder = ResNet18Encoder(freeze=True)  # start frozen, freeze_utils adjusts below
    elif args.encoder_backbone == "efficientnet_b0":
        encoder = EfficientNetB0Encoder(freeze=True)
    else:
        raise ValueError(f"Unknown encoder_backbone: {args.encoder_backbone}")

    apply_freeze_config(encoder, args.encoder_backbone, args.freeze_encoder)
    print(f"Encoder trainable params: {count_trainable_params(encoder)} "
          f"(freeze_encoder={args.freeze_encoder})")

    model = DefectLensAE(encoder=encoder, latent_channels=args.latent_channels)
    return model.to(device)


def build_loss(args):
    if args.loss == "mse":
        return MSELoss()
    elif args.loss == "composite":
        return CompositeLoss(alpha=args.alpha, beta=args.beta, win_size=args.ssim_win_size)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


@torch.no_grad()
def evaluate(model, dataloader, device, aggregation="max"):
    """Returns (avg_mse, auc_roc) for a val or test dataloader.
    avg_mse always uses plain MSE (regardless of training loss), so it's
    a consistent sanity signal to compare across ablations that use
    different training losses."""
    import numpy as np
    model.eval()
    all_scores, all_labels = [], []
    total_mse, n = 0.0, 0

    for batch in dataloader:
        x_encoder = batch["x_encoder"].to(device)
        x_target = batch["x_target"].to(device)

        recon = model(x_encoder)
        error_map = compute_pixel_error_map(recon, x_target)
        scores = aggregate_score(error_map, method=aggregation)

        all_scores.append(scores.cpu().numpy())
        all_labels.append(batch["label"].numpy())

        total_mse += ((recon - x_target) ** 2).mean().item() * x_encoder.size(0)
        n += x_encoder.size(0)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return total_mse / n, compute_auc_roc(scores, labels)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = get_datasets(
        args.data_root, args.category, args.splits_dir, augment_train=args.augment
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"[{args.category}] train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    model = build_model(args, device)
    criterion = build_loss(args)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = build_scheduler(optimizer, scheduler_type=args.scheduler,
                                 epochs=args.epochs, patience=args.scheduler_patience)

    if args.use_wandb:
        import wandb
        wandb.init(project="defectlens", name=args.run_name, config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_best.pt")
    best_val_auc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss, n_train = 0.0, 0

        for batch in train_loader:
            x_encoder = batch["x_encoder"].to(device)
            x_target = batch["x_target"].to(device)

            optimizer.zero_grad()
            recon = model(x_encoder)

            if args.loss == "composite":
                loss, _components = criterion(recon, x_target)
            else:
                loss = criterion(recon, x_target)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x_encoder.size(0)
            n_train += x_encoder.size(0)

        avg_train_loss = total_train_loss / n_train
        val_mse, val_auc = evaluate(model, val_loader, device, aggregation=args.score_aggregation)
        step_scheduler(scheduler, args.scheduler, val_auc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} | train_loss: {avg_train_loss:.4f} "
              f"| val_mse: {val_mse:.4f} | val_auc: {val_auc:.4f} | lr: {current_lr:.2e}")

        history.append({"epoch": epoch, "train_loss": avg_train_loss,
                         "val_mse": val_mse, "val_auc": val_auc, "lr": current_lr})

        if args.use_wandb:
            wandb.log({"train_loss": avg_train_loss, "val_mse": val_mse,
                       "val_auc": val_auc, "lr": current_lr, "epoch": epoch})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_auc": val_auc, "val_mse": val_mse, "args": vars(args),
            }, ckpt_path)
            print(f"  -> New best val_auc: {val_auc:.4f}, checkpoint saved.")

    history_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val_auc: {best_val_auc:.4f}. Checkpoint: {ckpt_path}")
    if args.use_wandb:
        wandb.finish()

    return ckpt_path, best_val_auc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", type=str, required=True, choices=["screw", "pill"])
    p.add_argument("--data_root", type=str, default="/content/mvtec_raw")
    p.add_argument("--splits_dir", type=str, default="/content/drive/MyDrive/defectlens/splits")
    p.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/defectlens/checkpoints")

    p.add_argument("--loss", type=str, default="mse", choices=["mse", "composite"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--ssim_win_size", type=int, default=11)

    p.add_argument("--encoder_backbone", type=str, default="resnet18",
                    choices=["resnet18", "efficientnet_b0"])
    p.add_argument("--freeze_encoder", action="store_true", default=True)
    p.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false")
    p.add_argument("--latent_channels", type=int, default=256)

    p.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "none"])
    p.add_argument("--scheduler_patience", type=int, default=5)

    p.add_argument("--score_aggregation", type=str, default="max", choices=["max", "mean"])
    p.add_argument("--augment", action="store_true", default=False)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--run_name", type=str, default="baseline_mse")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)