"""
src/eval/evaluate.py

Proper test-set evaluation for a trained checkpoint. This is the script
that produces the actual numbers that go in your results table -- run
once per (category, ablation) combination, after training is fully done,
on the held-out test set the model never influenced (not even indirectly
via checkpoint selection, which used val only).

What this does, and why each piece is here:
1. Loads a checkpoint, rebuilds the exact model architecture it was
   trained with (from the saved args, not guessed).
2. Runs the FULL test set through the model once.
3. Computes overall AUC-ROC under BOTH score aggregation methods (max
   and mean) -- not just whichever was used during training -- so you
   can see empirically whether the aggregation choice actually mattered
   on real data, not just the synthetic case in metrics.py.
4. Computes per-defect-type AUC-ROC breakdown.
5. Saves a handful of qualitative heatmap examples (best-detected and
   worst-detected defective images, plus a couple of good images) using
   visualize.py, so you have concrete visual evidence alongside the
   numbers -- useful both for sanity-checking the model and for your
   write-up.
6. Writes everything to a single JSON per run, so results across ablations
   can be collected into one table later without re-running evaluation.

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
from src.eval.visualize import plot_map


def rebuild_model_from_checkpoint(ckpt, device):
    """Rebuilds the exact architecture the checkpoint was trained with,
    using the args saved inside it -- never guessed or re-typed, so
    there's no risk of evaluating with a mismatched architecture."""
    saved_args = ckpt["args"]

    if saved_args["encoder_backbone"] == "resnet18":
        encoder = ResNet18Encoder(freeze=True)  # freeze state irrelevant at eval time
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
    """Single pass over the test set, collecting everything needed for
    both aggregation methods and per-defect-type breakdowns, plus raw
    tensors for a few examples to visualize -- avoids re-running the
    model multiple times for different analyses."""
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

    error_maps = torch.cat(all_error_maps, dim=0)      # [N, H, W]
    x_targets = torch.cat(all_x_target, dim=0)          # [N, 3, H, W]
    recons = torch.cat(all_recon, dim=0)                # [N, 3, H, W]
    labels = np.concatenate(all_labels)                 # [N]

    return error_maps, x_targets, recons, labels, all_defect_types, all_paths


def evaluate_checkpoint(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # weights_only=False: safe here since this is our own checkpoint, saved
    # by our own train.py, not a downloaded/untrusted file. PyTorch 2.6+
    # defaults to weights_only=True, which blocks loading the args/history
    # dict alongside the model weights.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, saved_args = rebuild_model_from_checkpoint(ckpt, device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_auc at save time: {ckpt['val_auc']:.4f})")
    print(f"Config: loss={saved_args['loss']}, freeze_encoder={saved_args['freeze_encoder']}, "
          f"encoder_backbone={saved_args['encoder_backbone']}")

    # Rebuild test set using the SAME data_root/splits_dir/category the
    # model was trained with -- ensures we're evaluating on the exact
    # held-out split, not accidentally a freshly-reshuffled one.
    _, _, test_ds = get_datasets(
        saved_args["data_root"], saved_args["category"], saved_args["splits_dir"]
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Test set size: {len(test_ds)}")

    error_maps, x_targets, recons, labels, defect_types, paths = run_full_test_eval(
        model, test_loader, device
    )

    # --- Both aggregation methods, on the SAME error maps (fair comparison) ---
    results = {"checkpoint": args.checkpoint, "config": saved_args, "test_set_size": len(test_ds)}

    for agg_method in ["max", "mean"]:
        scores = aggregate_score(error_maps, method=agg_method).numpy()
        overall_auc = compute_auc_roc(scores, labels)
        per_defect_auc = compute_per_defect_type_auc(scores, labels, defect_types)

        results[f"auc_{agg_method}"] = overall_auc
        results[f"per_defect_type_auc_{agg_method}"] = per_defect_auc

        print(f"\n--- Aggregation: {agg_method} ---")
        print(f"Overall test AUC-ROC: {overall_auc:.4f}")
        print("Per-defect-type AUC-ROC:")
        for dtype, auc in sorted(per_defect_auc.items(), key=lambda x: x[1]):
            print(f"  {dtype}: {auc:.4f}")

    # --- Save results JSON ---
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = os.path.basename(args.checkpoint).replace("_best.pt", "")
    results_path = os.path.join(args.output_dir, f"{run_name}_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Qualitative examples: best-detected defect, worst-detected
    # defect (most informative failure case), and a couple of good images ---
    if args.save_visuals:
        primary_scores = aggregate_score(error_maps, method=args.primary_aggregation).numpy()
        defect_mask = labels == 1
        good_mask = labels == 0

        defect_indices = np.where(defect_mask)[0]
        defect_scores_only = primary_scores[defect_mask]
        best_detected_idx = defect_indices[np.argmax(defect_scores_only)]   # highest score = correctly flagged strongly
        worst_detected_idx = defect_indices[np.argmin(defect_scores_only)]  # lowest score = model missed it

        good_indices = np.where(good_mask)[0]
        example_good_idx = good_indices[0] if len(good_indices) > 0 else None

        examples = {
            "best_detected_defect": best_detected_idx,
            "worst_detected_defect (likely missed)": worst_detected_idx,
        }
        if example_good_idx is not None:
            examples["example_good_image"] = example_good_idx

        for label_str, idx in examples.items():
            print(f"\nVisualizing: {label_str} | path: {paths[idx]} | "
                  f"defect_type: {defect_types[idx]} | score: {primary_scores[idx]:.4f}")
            plot_map(
                x_targets[idx], recons[idx], error_maps[idx],
                title=f"{label_str} ({defect_types[idx]}, score={primary_scores[idx]:.4f})"
            )

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/defectlens/results")
    p.add_argument("--save_visuals", action="store_true", default=True)
    p.add_argument("--primary_aggregation", type=str, default="max", choices=["max", "mean"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(args)