"""
src/eval/evaluate.py

Test-set evaluation for a trained checkpoint: overall + per-defect-type
AUC-ROC under both max and mean aggregation, a results JSON, and saved
PNG figures (examples, per-defect gallery, score histograms) for the
write-up / presentation.

Usage:
    python -m src.eval.evaluate --checkpoint /content/drive/MyDrive/defectlens/checkpoints/baseline_mse_screw_best.pt
"""

import os
import json
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import get_datasets
from src.models.encoder import ResNet18Encoder, EfficientNetB0Encoder
from src.models.autoencoder import DefectLensAE
from src.eval.metrics import (
    compute_pixel_error_map,
    aggregate_score,
    compute_auc_roc,
    compute_per_defect_type_auc,
)
from src.eval.visualize import plot_map, plot_defect_gallery, plot_score_distribution


def rebuild_model_from_checkpoint(ckpt, device):
    saved_args = ckpt["args"]

    if saved_args["encoder_backbone"] == "resnet18":
        encoder = ResNet18Encoder(freeze=True)
    elif saved_args["encoder_backbone"] == "efficientnet_b0":
        encoder = EfficientNetB0Encoder(freeze=True)
    else:
        raise ValueError(f"Unknown encoder_backbone in checkpoint: {saved_args['encoder_backbone']}")

    model = DefectLensAE(encoder=encoder, latent_channels=saved_args["latent_channels"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, saved_args


@torch.no_grad()
def run_full_test_eval(model, test_loader, device):
    all_error_maps, all_labels, all_defect_types = [], [], []
    all_x_target, all_recon, all_paths = [], [], []

    for batch in test_loader:
        x_encoder = batch["x_encoder"].to(device)
        x_target = batch["x_target"].to(device)

        recon = model(x_encoder)
        error_map = compute_pixel_error_map(recon, x_target)  # [B, H, W]

        all_error_maps.append(error_map.cpu())
        all_labels.append(batch["label"].numpy())
        all_defect_types.extend(batch["defect_type"])
        all_x_target.append(x_target.cpu())
        all_recon.append(recon.cpu())
        all_paths.extend(batch["path"])

    error_maps = torch.cat(all_error_maps, dim=0)
    x_targets = torch.cat(all_x_target, dim=0)
    recons = torch.cat(all_recon, dim=0)
    labels = np.concatenate(all_labels)

    return error_maps, x_targets, recons, labels, all_defect_types, all_paths


def evaluate_checkpoint(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, saved_args = rebuild_model_from_checkpoint(ckpt, device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_auc at save time: {ckpt['val_auc']:.4f})")
    print(f"Config: loss={saved_args['loss']}, freeze_encoder={saved_args['freeze_encoder']}, "
          f"encoder_backbone={saved_args['encoder_backbone']}")

    _, _, test_ds = get_datasets(
        saved_args["data_root"], saved_args["category"], saved_args["splits_dir"]
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Test set size: {len(test_ds)}")

    error_maps, x_targets, recons, labels, defect_types, paths = run_full_test_eval(
        model, test_loader, device
    )

    results = {"checkpoint": args.checkpoint, "config": saved_args, "test_set_size": len(test_ds)}
    all_scores = {}

    for agg_method in ["max", "mean"]:
        scores = aggregate_score(error_maps, method=agg_method).numpy()
        all_scores[agg_method] = scores
        overall_auc = compute_auc_roc(scores, labels)
        per_defect_auc = compute_per_defect_type_auc(scores, labels, defect_types)

        results[f"auc_{agg_method}"] = overall_auc
        results[f"per_defect_type_auc_{agg_method}"] = per_defect_auc

        print(f"\n--- Aggregation: {agg_method} ---")
        print(f"Overall test AUC-ROC: {overall_auc:.4f}")
        print("Per-defect-type AUC-ROC:")
        for dtype, auc in sorted(per_defect_auc.items(), key=lambda x: x[1]):
            print(f"  {dtype}: {auc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = os.path.basename(args.checkpoint).replace("_best.pt", "")
    results_path = os.path.join(args.output_dir, f"{run_name}_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Figures: saved as PNGs (plt.show() is a no-op in a subprocess) ---
    if args.save_visuals:
        fig_dir = os.path.join(args.output_dir, "figures", run_name)
        os.makedirs(fig_dir, exist_ok=True)
        vmax = float(error_maps.max())  # shared heatmap scale across all images

        # 1) Score histograms for both aggregations
        for agg in ["max", "mean"]:
            p = plot_score_distribution(
                all_scores[agg], labels,
                os.path.join(fig_dir, f"scores_{agg}.png"),
                title=f"{run_name} - {agg} aggregation (AUC={results[f'auc_{agg}']:.3f})",
            )
            print(f"Saved {p}")

        # 2) Per-defect-type gallery + best/worst examples, per aggregation
        for agg in ["max", "mean"]:
            sc = all_scores[agg]
            p = plot_defect_gallery(
                x_targets, recons, error_maps, sc, labels, defect_types,
                os.path.join(fig_dir, f"gallery_{agg}.png"), vmax=vmax,
                title=f"{run_name} - top-scoring image per defect type ({agg})",
            )
            print(f"Saved {p}")

            defect_idx = np.where(labels == 1)[0]
            good_idx = np.where(labels == 0)[0]
            examples = {
                "best_detected": defect_idx[np.argmax(sc[defect_idx])],
                "worst_detected": defect_idx[np.argmin(sc[defect_idx])],
            }
            if len(good_idx):
                examples["good_example"] = good_idx[0]
            for name, idx in examples.items():
                p = plot_map(
                    x_targets[idx], recons[idx], error_maps[idx],
                    title=f"{name} ({defect_types[idx]}, {agg} score={sc[idx]:.4f})",
                    save_path=os.path.join(fig_dir, f"{name}_{agg}.png"), vmax=vmax,
                )
                print(f"Saved {p}")

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/defectlens/results")
    p.add_argument("--save_visuals", action="store_true", default=True)
    p.add_argument("--no_save_visuals", dest="save_visuals", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(args)