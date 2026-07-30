"""
dataset.py

Handles data loading for DefectLens (MVTec AD - screw, pill categories).

Responsibilities:
1. Enumerate train/good, and stratify test/* into a fixed val/test split.
2. Save/load that split so it's identical across every run (Step 3 decision:
   fixed-and-saved, so ablation runs are compared on the same yardstick).
3. Provide a PyTorch Dataset that returns BOTH:
   - encoder input: ImageNet-normalized tensor
   - reconstruction target: plain [0,1] tensor
   (Step 3 verify condition: dual normalization must not be conflated.)
4. Provide labels (0=good, 1=defective) + defect_type string for val/test,
   so later per-defect-type AUC-ROC breakdowns are possible.

Small defect-type folders (< MIN_STRATIFY_COUNT images) are excluded from
val stratification entirely -- they go straight to the final test set,
since splitting e.g. 9 images meaningfully isn't possible. This is a
judgment call, documented here rather than left implicit.
"""

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ---- Constants -------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_SIZE = 256
VAL_FRACTION = 0.15          # ~15% of test/* stratified into val
MIN_STRATIFY_COUNT = 15      # defect folders smaller than this skip val entirely
SEED = 42


# ---- Split creation ---------------------------------------------------

def build_split(data_root: str, category: str, splits_dir: str, seed: int = SEED):
    """
    Builds (or loads, if already saved) a fixed stratified val/test split
    for a given category (e.g. 'screw', 'pill').

    Returns a dict:
        {
          "val":  [{"path": ..., "label": 0/1, "defect_type": "good"/"scratch"/...}, ...],
          "test": [...]
        }
    and writes it to {splits_dir}/{category}_val_test_split.json so re-running
    the notebook never silently reshuffles which images are val vs test.
    """
    os.makedirs(splits_dir, exist_ok=True)
    split_path = Path(splits_dir) / f"{category}_val_test_split.json"

    if split_path.exists():
        with open(split_path, "r") as f:
            return json.load(f)

    rng = random.Random(seed)
    test_root = Path(data_root) / category / "test"
    defect_types = sorted(os.listdir(test_root))  # includes "good"

    val_records, test_records = [], []

    for defect_type in defect_types:
        folder = test_root / defect_type
        files = sorted(os.listdir(folder))
        label = 0 if defect_type == "good" else 1

        records = [
            {"path": str(folder / fname), "label": label, "defect_type": defect_type}
            for fname in files
        ]

        # Small pools (e.g. pill_type with 9 images) aren't stratified into
        # val -- they go entirely to final test. Splitting them further
        # would leave too few images on either side to mean anything.
        if len(records) < MIN_STRATIFY_COUNT:
            test_records.extend(records)
            continue

        rng.shuffle(records)
        n_val = max(1, round(len(records) * VAL_FRACTION))
        val_records.extend(records[:n_val])
        test_records.extend(records[n_val:])

    split = {"val": val_records, "test": test_records}
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    return split


# ---- Datasets -----------------------------------------------------------

def _build_transforms():
    """Returns the two transforms applied to the SAME resized image:
    one ImageNet-normalized (encoder input), one plain [0,1] (recon target)."""
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    to_tensor = transforms.ToTensor()  # -> [0,1]
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return resize, to_tensor, normalize


class MVTecTrainDataset(Dataset):
    """Train split: train/good only. No labels needed -- reconstruction
    target IS the input, just in a different normalization."""

    def __init__(self, data_root: str, category: str, augment: bool = False):
        self.root = Path(data_root) / category / "train" / "good"
        self.files = sorted(os.listdir(self.root))
        self.resize, self.to_tensor, self.normalize = _build_transforms()
        self.augment = augment
        if augment:
            self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.root / self.files[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.resize(img)
        if self.augment:
            img = self.aug(img)

        x_target = self.to_tensor(img)             # [0,1], reconstruction target
        x_encoder = self.normalize(x_target.clone())  # ImageNet-normalized, encoder input

        return {"x_encoder": x_encoder, "x_target": x_target}


class MVTecEvalDataset(Dataset):
    """Val or test split: mix of good + defective images, with labels.
    Used for AUC-ROC computation (Val Set B from our discussion), not
    just loss monitoring."""

    def __init__(self, records: list):
        self.records = records
        self.resize, self.to_tensor, self.normalize = _build_transforms()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        img = self.resize(img)

        x_target = self.to_tensor(img)
        x_encoder = self.normalize(x_target.clone())

        return {
            "x_encoder": x_encoder,
            "x_target": x_target,
            "label": rec["label"],            # 0 = good, 1 = defective
            "defect_type": rec["defect_type"],
            "path": rec["path"],
        }


# ---- Convenience builder -------------------------------------------------

def get_datasets(data_root: str, category: str, splits_dir: str, augment_train: bool = False):
    """One call to get train/val/test datasets for a category, with the
    split fixed and saved on first call."""
    split = build_split(data_root, category, splits_dir)
    train_ds = MVTecTrainDataset(data_root, category, augment=augment_train)
    val_ds = MVTecEvalDataset(split["val"])
    test_ds = MVTecEvalDataset(split["test"])
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Quick sanity check -- run this directly to verify shapes/normalization
    # before wiring into the training loop (Step 3's verify condition).
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "/content/mvtec_raw"
    category = sys.argv[2] if len(sys.argv) > 2 else "screw"
    splits_dir = sys.argv[3] if len(sys.argv) > 3 else "/content/drive/MyDrive/defectlens/splits"

    train_ds, val_ds, test_ds = get_datasets(data_root, category, splits_dir)
    print(f"[{category}] train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    sample = train_ds[0]
    print("x_encoder shape:", sample["x_encoder"].shape,
          "mean:", sample["x_encoder"].mean().item())
    print("x_target shape:", sample["x_target"].shape,
          "min/max:", sample["x_target"].min().item(), sample["x_target"].max().item())

    eval_sample = val_ds[0]
    print("val sample label:", eval_sample["label"], "defect_type:", eval_sample["defect_type"])