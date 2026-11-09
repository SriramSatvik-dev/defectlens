"""
src/eval/visualize.py

Plotting helpers for DefectLens. Every function SAVES a PNG (when given a
save_path) instead of relying on plt.show(), because evaluate.py runs in a
`!python -m ...` subprocess where matplotlib has no interactive backend and
plt.show() silently draws nothing. Saved PNGs are displayed from the
notebook instead (see the notebook cell in the project notes).
"""

import os

import matplotlib
matplotlib.use("Agg")  # explicit: headless-safe, we always save to disk
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_img(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_map(x_target, recon, error_map, title="Map Visualization",
             save_path=None, vmax=None):
    """Target | Reconstruction | Error map for ONE image.
    x_target, recon: [3, H, W] in [0,1]; error_map: [H, W].
    vmax: shared upper bound for the error colormap so heatmaps are
    comparable across images (pass the test-set max error)."""
    err = error_map.detach().cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(_to_img(x_target)); axs[0].set_title("Target")
    axs[1].imshow(_to_img(recon)); axs[1].set_title("Reconstruction")
    im = axs[2].imshow(err, cmap="hot", vmin=0, vmax=vmax)
    axs[2].set_title("Error map")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    for a in axs:
        a.axis("off")

    fig.suptitle(title)
    _save(fig, save_path)
    return save_path


def plot_defect_gallery(x_targets, recons, error_maps, scores, labels,
                        defect_types, save_path, vmax=None, title="Gallery"):
    """One row per defect type (its highest-scoring image) plus one 'good'
    row. Columns: target | reconstruction | error map. Best single slide
    for showing what the model catches."""
    defect_types = np.array(defect_types)
    rows = []
    good_idx = np.where(labels == 0)[0]
    if len(good_idx):
        rows.append(("good", good_idx[np.argmin(scores[good_idx])]))
    for dt in sorted(set(defect_types[labels == 1])):
        idx = np.where(defect_types == dt)[0]
        rows.append((dt, idx[np.argmax(scores[idx])]))

    fig, axs = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    axs = np.atleast_2d(axs)
    for r, (name, i) in enumerate(rows):
        axs[r, 0].imshow(_to_img(x_targets[i]))
        axs[r, 1].imshow(_to_img(recons[i]))
        axs[r, 2].imshow(error_maps[i].numpy(), cmap="hot", vmin=0, vmax=vmax)
        axs[r, 0].set_ylabel(f"{name}\nscore={scores[i]:.3f}", fontsize=10)
        for c in range(3):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    for c, t in enumerate(["Target", "Reconstruction", "Error map"]):
        axs[0, c].set_title(t)

    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, save_path)
    return save_path


def plot_score_distribution(scores, labels, save_path, title="Score distribution"):
    """Histogram of anomaly scores, good vs defective. Shows separation
    (or lack of it) at a glance."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(scores.min(), scores.max(), 35)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.65, label="Good", color="tab:blue")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.65, label="Defective", color="tab:red")
    ax.set_xlabel("Anomaly score"); ax.set_ylabel("Number of images")
    ax.set_title(title); ax.legend()
    _save(fig, save_path)
    return save_path