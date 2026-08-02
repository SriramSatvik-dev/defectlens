"""
eval/metrics.py

Turns per-pixel reconstruction error into per-image anomaly scores, and
evaluates those scores against ground-truth labels (AUC-ROC).

Score aggregation (Step 10's decision, made concrete here):
- "max": the single worst-reconstructed pixel's error becomes the image's
  score. More sensitive to small, localized defects (most of screw/pill's
  defect types) but noisier -- a single outlier pixel (e.g. a compression
  artifact) can spike the score.
- "mean": average error across all pixels. Smoother/more stable, but can
  under-detect small defects, since a tiny bad region gets diluted by a
  large well-reconstructed area.
Both are implemented; which one is "the" model's score is decided
empirically by comparing AUC-ROC under each, not assumed in advance.

This file deliberately keeps aggregation as an explicit function argument
everywhere, rather than a hardcoded choice, so that comparison is a single
function-call away, not a code change.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_pixel_error_map(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel reconstruction error, averaged across the 3 color channels.
    Input: recon, target -- [B, 3, H, W], both in [0,1]
    Output: [B, H, W] error map
    """
    return (recon - target).abs().mean(dim=1)  # [B, H, W]


def aggregate_score(error_map: torch.Tensor, method: str = "max") -> torch.Tensor:
    """Collapses a per-pixel error map into one scalar anomaly score per image.
    Input: error_map [B, H, W]
    Output: [B] scores
    """
    flat = error_map.flatten(start_dim=1)  # [B, H*W]
    if method == "max":
        return flat.max(dim=1).values
    elif method == "mean":
        return flat.mean(dim=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Use 'max' or 'mean'.")


@torch.no_grad()
def compute_scores_and_labels(model, dataloader, device, aggregation: str = "max"):
    """Runs the model over a dataloader (val or test split), returns:
    - scores: np.array of per-image anomaly scores
    - labels: np.array of ground-truth labels (0=good, 1=defective)
    - defect_types: list of defect_type strings, aligned with scores/labels

    Expects dataloader to yield dicts matching MVTecEvalDataset's output
    (x_encoder, x_target, label, defect_type).
    """
    model.eval()
    all_scores, all_labels, all_defect_types = [], [], []

    for batch in dataloader:
        x_encoder = batch["x_encoder"].to(device)
        x_target = batch["x_target"].to(device)

        recon = model(x_encoder)
        error_map = compute_pixel_error_map(recon, x_target)
        scores = aggregate_score(error_map, method=aggregation)

        all_scores.append(scores.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_defect_types.extend(batch["defect_type"])

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return scores, labels, all_defect_types


def compute_auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standard AUC-ROC: how well do anomaly scores separate good (0)
    from defective (1) images, across all possible thresholds."""
    return roc_auc_score(labels, scores)


def compute_per_defect_type_auc(scores: np.ndarray, labels: np.ndarray, defect_types: list) -> dict:
    """Breaks AUC-ROC down per defect type, comparing each defect type's
    scores against the 'good' images' scores only (standard practice --
    otherwise a defect type with few samples gets diluted by other,
    possibly easier, defect types in the same overall AUC number)."""
    defect_types = np.array(defect_types)
    good_mask = labels == 0
    good_scores = scores[good_mask]

    results = {}
    for dtype in sorted(set(defect_types[labels == 1])):
        dtype_mask = defect_types == dtype
        dtype_scores = scores[dtype_mask]
        combined_scores = np.concatenate([good_scores, dtype_scores])
        combined_labels = np.concatenate([np.zeros(len(good_scores)), np.ones(len(dtype_scores))])
        results[dtype] = roc_auc_score(combined_labels, combined_scores)
    return results


if __name__ == "__main__":
    # Verify condition: aggregation + AUC computation work correctly on
    # synthetic data where the "correct" answer is known in advance.
    torch.manual_seed(0)

    # Synthetic error maps: good images have low, uniform error;
    # defective images have one bright localized spike.
    B, H, W = 10, 16, 16
    error_maps = torch.rand(B, H, W) * 0.1  # baseline low error for all
    labels = np.array([0]*5 + [1]*5)
    for i in range(5, 10):  # defective images get one localized spike
        error_maps[i, 8, 8] = 0.9

    max_scores = aggregate_score(error_maps, method="max").numpy()
    mean_scores = aggregate_score(error_maps, method="mean").numpy()

    print("Max scores:", max_scores)
    print("Mean scores:", mean_scores)

    max_auc = compute_auc_roc(max_scores, labels)
    mean_auc = compute_auc_roc(mean_scores, labels)
    print(f"AUC-ROC (max aggregation): {max_auc:.4f} (expect ~1.0 -- spike is very localized)")
    print(f"AUC-ROC (mean aggregation): {mean_auc:.4f} (expect lower than max -- spike diluted by averaging)")

    assert max_auc > mean_auc, "Max aggregation should outperform mean on a localized-spike synthetic case"
    print("Sanity check passed: max aggregation correctly more sensitive to localized defects here.")