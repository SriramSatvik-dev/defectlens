# DefectLens

Unsupervised, reconstruction-based anomaly detection for manufacturing defect detection, built on the MVTec AD benchmark (`screw` and `pill` categories).

A pretrained ResNet18 (truncated at `layer3`) encodes an image into a spatial feature map; a custom `ConvTranspose2d` decoder reconstructs it. Trained only on defect-free images, the model learns what "normal" looks like — at inference, wherever a reconstruction disagrees with its input is the anomaly signal. No labeled defects are used at training time. EfficientNet-B0 is available as an alternative encoder for the backbone-comparison ablation.

## Why this approach

Real manufacturing defects are rare and varied — a supervised classifier only ever recognizes the specific defect types it was shown during training. Reconstruction-based detection sidesteps this: it only needs to learn what "good" looks like, so it generalizes to defect types it has never seen. See [`ENGINEERING_DECISIONS.md`](./ENGINEERING_DECISIONS.md) for the full reasoning and every other design decision made in this project.

## Architecture

```
Input [3, 256, 256]
  → Encoder (pretrained ResNet18, truncated at layer3, frozen by default) → [256, 16, 16]
  → Bottleneck (1×1 conv, tunable latent_channels)                        → [latent_ch, 16, 16]
  → Decoder (custom ConvTranspose2d stack, Sigmoid output)                → [3, 256, 256]
```

The encoder is swappable with `--encoder_backbone`. The ablation backbone, EfficientNet-B0, is truncated to the same /16 downsampling but outputs 112 channels instead of 256; the 1×1 bottleneck absorbs the difference, so the decoder is unchanged.

Anomaly score = aggregated (max or mean) per-pixel reconstruction error, evaluated via AUC-ROC — a continuous ranking metric, not a fixed good/defective classifier (see Scope Notes in [`ENGINEERING_DECISIONS.md`](./ENGINEERING_DECISIONS.md)).

## Results

| Category | Best config | max AUC | mean AUC |
|---|---|---|---|
| **Screw** | Composite loss (L1+SSIM) + fine-tuned ResNet18 | 0.8727 | **1.0000** |
| **Pill** | Baseline (MSE, frozen ResNet18) | 0.6947 | **0.8826** |

Full comparison table across every ablation: [`results.md`](./results.md).

**Headline finding:** no single configuration wins for both categories. Composite loss and fine-tuning substantially improve screw (structural, localized defects) but actively degrade pill (color/intensity-based defects) below its own MSE baseline — evidence and mechanism in [`ENGINEERING_DECISIONS.md`](./ENGINEERING_DECISIONS.md).

### Backbone ablation (EfficientNet-B0 vs ResNet18)

Both runs use plain MSE and a frozen encoder, so the backbone is the only change.

| Category | Backbone | max AUC | mean AUC |
|---|---|---|---|
| Screw | ResNet18 (256 ch) | 0.8436 | 0.0141 |
| Screw | EfficientNet-B0 (112 ch) | 0.8286 | 0.0054 |
| Pill | ResNet18 (256 ch) | 0.6947 | 0.8826 |
| Pill | EfficientNet-B0 (112 ch) | 0.6970 | 0.8034 |

Swapping the backbone barely moved screw, which reproduced ResNet18's near-total mean-aggregation inversion. The one clear cost was on pill's `color` defect type (mean AUC 0.87 → 0.67), where the narrower 112-channel feature map appears to be the limiting factor. Loss function and defect type mattered far more than backbone choice. Stacking composite loss, fine-tuning and EfficientNet-B0 together did not beat the best single-lever config for either category (screw 0.8000 max / 1.0000 mean; pill 0.4924 max / 0.7697 mean).

### Score distributions (good vs defective)

Histograms of per-image anomaly scores on the test set, for the best config of each category. The less the red (defective) and blue (good) distributions overlap, the better the separation.

**Screw** — composite loss + fine-tuned ResNet18

| Mean aggregation | Max aggregation |
|---|---|
| ![Screw score distribution, mean aggregation](docs/images/screw_scores_mean.png) | ![Screw score distribution, max aggregation](docs/images/screw_scores_max.png) |

**Pill** — baseline (MSE, frozen ResNet18)

| Mean aggregation | Max aggregation |
|---|---|
| ![Pill score distribution, mean aggregation](docs/images/pill_scores_mean.png) | ![Pill score distribution, max aggregation](docs/images/pill_scores_max.png) |

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
│   ├── evaluate.py         # full test-set evaluation, per-checkpoint, saves figures
│   └── visualize.py        # saved plots: heatmaps, per-defect gallery, score histograms
notebooks/
└── collab_runner.ipynb     # Drive mount, repo pull, entrypoint calls
docs/
└── images/                 # histograms embedded in this README
results.md                  # full ablation results table
ENGINEERING_DECISIONS.md    # design decisions + empirical findings, with evidence
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

# Same, with the EfficientNet-B0 backbone
python -m src.training.train --category screw --loss mse --encoder_backbone efficientnet_b0 --lr 1e-3 --run_name my_effnet_run

# Evaluate a checkpoint on the held-out test set
python -m src.eval.evaluate --checkpoint /path/to/my_run_best.pt
```

Checkpoints, val/test splits, and results are persisted to Google Drive across sessions; the raw MVTec AD dataset (`screw`, `pill` categories) is expected at a local path passed via `--data_root`.

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — `screw` (320 train / 160 test) and `pill` (267 train / 167 test) categories.