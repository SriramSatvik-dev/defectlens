# DefectLens

Unsupervised, reconstruction-based anomaly detection for manufacturing defect detection, built on the MVTec AD benchmark (`screw` and `pill` categories).

A pretrained ResNet18 (truncated at `layer3`) encodes an image into a spatial feature map; a custom `ConvTranspose2d` decoder reconstructs it. Trained only on defect-free images, the model learns what "normal" looks like — at inference, wherever a reconstruction disagrees with its input is the anomaly signal. No labeled defects are used at training time.

## Why this approach

Real manufacturing defects are rare and varied — a supervised classifier only ever recognizes the specific defect types it was shown during training. Reconstruction-based detection sidesteps this: it only needs to learn what "good" looks like, so it generalizes to defect types it has never seen. See [`decisions.md`](./decisions.md) for the full reasoning and every other design decision made in this project.

## Architecture

```
Input [3, 256, 256]
  → Encoder (pretrained ResNet18, truncated at layer3, frozen by default) → [256, 16, 16]
  → Bottleneck (1×1 conv, tunable latent_channels)                        → [latent_ch, 16, 16]
  → Decoder (custom ConvTranspose2d stack, Sigmoid output)                → [3, 256, 256]
```

Anomaly score = aggregated (max or mean) per-pixel reconstruction error, evaluated via AUC-ROC — a continuous ranking metric, not a fixed good/defective classifier (see Scope Notes in [`decisions.md`](./decisions.md)).

## Results

| Category | Best config | max AUC | mean AUC |
|---|---|---|---|
| **Screw** | Composite loss (L1+SSIM) + fine-tuned ResNet18 | 0.8727 | **1.0000** |
| **Pill** | Baseline (MSE, frozen ResNet18) | 0.6947 | **0.8826** |

Full comparison table across every ablation: [`results.md`](./results.md).

**Headline finding:** no single configuration wins for both categories. Composite loss and fine-tuning substantially improve screw (structural, localized defects) but actively degrade pill (color/intensity-based defects) below its own MSE baseline — evidence and mechanism in [`decisions.md`](./decisions.md).

## Ablations run

- Baseline: MSE loss, frozen ResNet18
- Composite loss (L1 + SSIM)
- Partial fine-tuning (encoder `layer3` unfrozen)
- Backbone swap (EfficientNet-B0)
- Best-of-all-levers combined, per category

Each ablation is a config flag on the same training script (`src/training/train.py`), not a separate implementation — see repo structure below.

## Repo structure

```
src/
├── data/
│   ├── dataset.py          # stratified val/test split, dual normalization
│   └── transforms.py
├── models/
│   ├── encoder.py          # ResNet18 (primary) + EfficientNet-B0 (ablation)
│   ├── bottleneck.py       # tunable 1x1 conv
│   ├── decoder.py          # ConvTranspose2d stack
│   └── autoencoder.py      # wires the above together
├── losses/
│   ├── mse_loss.py
│   └── ssim_composite_loss.py
├── training/
│   ├── train.py            # main training loop, checkpoints on best val AUC-ROC
│   ├── freeze_utils.py     # encoder freeze / partial-unfreeze logic
│   └── scheduler.py        # LR scheduling
├── eval/
│   ├── metrics.py          # score aggregation (max/mean), AUC-ROC
│   ├── evaluate.py         # full test-set evaluation, per-checkpoint
│   └── visualize.py        # reconstruction + error heatmap plotting
notebooks/
└── colab_runner.ipynb      # Drive mount, repo pull, entrypoint calls
results.md                  # full ablation results table
decisions.md                # design decisions + empirical findings, with evidence
```

## Setup

```bash
pip install -r requirements.txt
```

(On Colab specifically, use `pip install -r requirements.txt --break-system-packages` if needed, and note `torch`/`torchvision` are usually already installed in the runtime image — only `pytorch-msssim` and `wandb` typically need adding.)

## Running it

Training and evaluation run on Colab (GPU); code is authored and versioned separately via this repo.

```bash
# Train (any config is a flag combination on the same script)
python -m src.training.train --category screw --loss composite --epochs 50 --lr 1e-4 --run_name my_run

# Evaluate a checkpoint on the held-out test set
python -m src.eval.evaluate --checkpoint /path/to/my_run_best.pt
```

Checkpoints, val/test splits, and results are persisted to Google Drive across sessions; the raw MVTec AD dataset (`screw`, `pill` categories) is expected at a local path passed via `--data_root`.

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — `screw` (320 train / 160 test) and `pill` (267 train / 167 test) categories.
