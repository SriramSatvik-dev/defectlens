# CLAUDE.md — MVTec AD Defect Detection: Custom Autoencoder with Pretrained Encoder

This file is the authoritative reference for any AI coding agent working on this
project. Read it fully before writing any code, creating any file, or running any
command. Every architectural decision in here was made deliberately — do not
deviate without flagging it first.

---

## 1. Project Overview

An unsupervised visual anomaly detection system for manufacturing defect
detection, built as a deep-learning-depth portfolio project. The model is
trained **only on defect-free ("good") images** per category from the MVTec AD
dataset, and learns to reconstruct normal patterns. At test time, defective
regions reconstruct poorly, and the pixel/structural reconstruction error map
becomes the anomaly signal — both for image-level detection (is this image
defective?) and pixel-level localization (where is the defect?).

**Why this project exists**: this is the third project in a three-project AI/ML
resume arc. The other two (FinQuery — RAG/retrieval systems, PhishGuard —
deployed applied ML) demonstrate systems and deployment depth but not hands-on
neural network training depth. This project exists specifically to prove
"I understand what's happening inside the model" via:
- A custom-written PyTorch training loop (no `Trainer.fit()` or wrappers)
- A deliberately designed composite loss function
- A hybrid architecture: pretrained encoder + fully custom decoder
- An explicit, ablated fine-tuning strategy (frozen → partial unfreeze)
- Rigorous, logged ablation studies

**Builder profile**: Final-year B.Tech student (CS), preparing for AI/ML
campus placements, working alongside DSA/OA prep. Strong software engineering
background. **No prior experience writing a custom PyTorch training loop** —
this is being learned as part of the project, not assumed. No local GPU
(AMD Ryzen 5 + Radeon, no CUDA) — all training happens on Google Colab free-tier
T4 GPU.

**Priority**: finishing with rigor over ambition. A smaller, fully-executed
project with real ablations beats a bigger idea that's half-finished. Compute-
hungry approaches (self-supervised pretraining, diffusion models) are explicitly
out of scope for this iteration.

---

## 2. Domain and Dataset

- **Dataset**: MVTec AD (MVTec Anomaly Detection), a standard, well-known,
  clean public benchmark for unsupervised defect detection.
- **Categories** (locked, exactly two — chosen for low risk):
  - **screw** — object category, consistent pose, mostly textural/local
    defects (scratches, thread damage). Reconstructs well with plain
    conv-autoencoders.
  - **pill** — object category, consistent pose, textural + color +
    contamination defects. Also known to give solid AUC-ROC with
    reconstruction-based methods.
- **Explicitly excluded for this iteration**:
  - **cable** — structural defects (missing/swapped wires) are a documented
    hard case for reconstruction-based autoencoders; the model often
    reconstructs the defect too well, which hurts anomaly scores. Optional
    stretch goal only, after core scope (both locked categories, full
    ablation suite) is complete.
  - **transistor** — pose/alignment variation adds a separate alignment
    problem outside the scope of what this project is testing (loss design,
    fine-tuning, ablations). Not planned at all for this iteration.
- **Why two categories, not one or four**: one category risks looking thin
  for a portfolio project; four categories (the original candidate list)
  risks debugging time ballooning across categories with different failure
  modes before a single baseline is proven. Two well-behaved categories give
  a genuine cross-category generalization result without that risk.

**Sequencing rule**: get the full pipeline (data loading → training loop →
loss → eval) working end-to-end on **screw only** first. Only after screw has
a working baseline and passes a sanity check should pill be added. Do not
develop both categories in parallel from day one.

---

## 3. Tech Stack

| Layer                  | Tool                                                |
|------------------------|------------------------------------------------------|
| Framework              | PyTorch                                              |
| Encoder (pretrained)   | ResNet18 (ImageNet-pretrained, torchvision), truncated at layer3 |
| Encoder (ablation-only)| EfficientNet-B0 (backbone comparison ablation)       |
| Decoder                | Custom-built, `ConvTranspose2d` stack (from scratch) |
| Loss (baseline)        | Plain MSE (naive baseline, for ablation comparison)  |
| Loss (main model)      | Composite: α·L1/MSE + β·(1 − SSIM)                   |
| Mixed precision        | `torch.cuda.amp` (autocast + GradScaler)             |
| Experiment tracking    | Weights & Biases (W&B) — scalars + config + sample images only, no sweeps/artifacts |
| Compute                | Google Colab, free-tier T4 GPU, accessed via VS Code |
| Checkpoint storage     | Google Drive (mounted) — required, since Colab disconnects on idle and local Colab disk is ephemeral |
| Data                   | MVTec AD, categories: screw, pill                    |

---

## 4. Repository Structure

```
defectlens/
│
├── CLAUDE.md                        # this file
├── README.md                        # project overview + demo instructions
├── requirements.txt
├── .gitignore
│
├── config/
│   └── config.yaml                  # hyperparams, paths, category, loss weights
│
├── data/
│   ├── raw/                         # downloaded MVTec AD (per category), gitignored
│   └── splits/                      # train (good only) / val (good) / test (good+defect) index files
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # MVTecDataset — dual-normalized image pairs
│   │   └── transforms.py            # resize, ImageNet-norm view, [0,1]-norm view
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py               # ResNet18 / EfficientNet-B0 truncated wrappers
│   │   ├── bottleneck.py            # 1x1 conv, tunable latent_channels
│   │   ├── decoder.py               # custom ConvTranspose2d decoder stack
│   │   └── autoencoder.py           # wires encoder + bottleneck + decoder together
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── mse_loss.py              # naive baseline
│   │   └── ssim_composite_loss.py   # alpha*L1/MSE + beta*(1-SSIM)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                 # custom training loop (forward, backward, AMP, ckpt)
│   │   ├── freeze_utils.py          # freeze/unfreeze logic for fine-tuning ablation
│   │   └── scheduler.py             # LR scheduling setup
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py               # image-level AUC-ROC, pixel-level metrics
│   │   └── visualize.py             # reconstruction + error heatmap grids
│   │
│   └── ablations/
│       ├── __init__.py
│       ├── run_ablation_loss.py         # MSE vs MSE+SSIM
│       ├── run_ablation_finetune.py     # frozen vs partial-unfreeze
│       ├── run_ablation_latent_dim.py   # latent_channels sweep
│       └── run_ablation_backbone.py     # ResNet18 vs EfficientNet-B0
│
├── notebooks/
│   └── colab_runner.ipynb           # thin notebook: mounts Drive, calls src/ entrypoints
│
└── results/
    ├── checkpoints/                 # best-val-metric checkpoints, saved to Drive
    ├── ablation_tables/             # CSV/markdown comparison tables per ablation
    └── figures/                     # reconstruction grids, error heatmaps, ROC curves
```

---

## 5. Architecture

### 5.1 Encoder

- **Primary**: `torchvision.models.resnet18(pretrained=True)`, truncated —
  drop `avgpool` and `fc`, keep through `layer3`.
- Input: 256×256×3, ImageNet-normalized (mean/std).
- Output: feature map of shape approximately **16×16×256**.
- **Ablation-only**: `EfficientNet-B0` (torchvision or `timm`), truncated to a
  comparable spatial resolution. Used solely for the backbone-comparison
  ablation (Section 7) — not the primary model.
- **Why ResNet18 over EfficientNet-B0 as primary**: smaller, faster to train
  on a T4, and its block-by-block structure (BasicBlock, residual connections)
  is easier to explain and defend in an interview than EfficientNet's compound
  scaling. EfficientNet is introduced later purely as a comparison point.

### 5.2 Bottleneck

```python
self.bottleneck = nn.Conv2d(256, latent_channels, kernel_size=1)
```
- `latent_channels` default: 256 (no compression) for the main model.
- This is the tunable parameter for the **latent-dimension ablation**
  (Section 7) — compare e.g. 256 / 128 / 64.

### 5.3 Decoder (fully custom, from scratch)

Mirrors the encoder's downsampling with symmetric upsampling via
`ConvTranspose2d`:

```python
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_channels=256, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # reconstruction target is [0,1]-normalized
        )

    def forward(self, x):
        return self.decoder(x)
```

- BatchNorm + ReLU after each transpose conv for training stability.
- Sigmoid output assumes the reconstruction **target** image is normalized
  to [0,1] — see Section 5.4 for why this requires dual normalization.
- **Known risk / documented fallback**: `ConvTranspose2d` can produce
  checkerboard artifacts. If observed, the documented fix is swapping to
  `nn.Upsample(scale_factor=2, mode="nearest") + Conv2d`. Treat this as an
  optional mini-ablation, not a required one.

### 5.4 Dual normalization (critical detail — do not silently get wrong)

The encoder (ResNet18) expects ImageNet mean/std-normalized input. The
decoder's Sigmoid output and the reconstruction loss expect the target image
in [0,1] range. This means the dataset must produce **two views of the same
image**:
- `x_encoder`: resized to 256×256, ImageNet-normalized — fed into the encoder.
- `x_target`: resized to 256×256, scaled to [0,1] only — used as the
  reconstruction target for the loss.

`src/data/dataset.py` must return both explicitly (e.g. as a dict or tuple),
never conflate them.

---

## 6. Loss Functions

### 6.1 Naive baseline: plain MSE

Per-pixel `MSELoss` between `x_target` and the reconstruction. Included
specifically as the "before" side of the loss ablation — not the final model.

**Why it's insufficient alone**: MSE is minimized by predicting the average of
plausible pixel values, producing blurry reconstructions. Blurry
reconstructions mean even normal images carry noticeable pixel-wise error
everywhere, which drowns out the localized error spike a real defect should
produce — hurting anomaly localization precision.

### 6.2 Main model: composite loss

```
L = alpha * L1(recon, target) + beta * (1 - SSIM(recon, target))
```

- SSIM (Structural Similarity) measures local structural similarity
  (luminance, contrast, structure), not raw pixel difference — it is far
  more sensitive to localized structural anomalies (scratches, dents) and is
  the standard choice in the MVTec/defect-detection autoencoder literature
  (e.g. Bergmann et al.'s SSIM-based autoencoder baselines).
- The L1/MSE term anchors overall reconstruction fidelity; the SSIM term
  sharpens structural sensitivity.
- Starting weights: `alpha=1.0`, `beta=1.0` — tune only if val metrics
  clearly demand it; document any change and why.
- This composite loss, and the MSE→composite comparison, is the required
  "deliberate, documented loss decision" for the project (not an off-the-shelf
  single loss with no reasoning).

---

## 7. Fixed Ablation Requirements (minimum 3–4, each logged as a comparison table)

1. **Loss function**: plain MSE vs. MSE(or L1)+SSIM composite. Compare
   image-level AUC-ROC and qualitative reconstruction sharpness.
2. **Fine-tuning strategy**: frozen encoder vs. partial unfreeze
   (`layer3`, and optionally `layer2`, unfrozen; `conv1`/`layer1` stay frozen
   — early layers capture generic low-level features that don't need
   retraining). Compare AUC-ROC and training stability.
3. **Latent dimension**: `latent_channels` in {256, 128, 64} via the
   bottleneck conv. Compare reconstruction quality vs. AUC-ROC.
4. **Backbone comparison**: ResNet18 vs. EfficientNet-B0 as the frozen
   encoder. Compare AUC-ROC vs. inference latency.

Each ablation must produce a quantitative comparison table (not just
qualitative impressions), saved under `results/ablation_tables/`.

---

## 8. Evaluation

- **Primary metric**: image-level AUC-ROC (standard for MVTec AD anomaly
  detection) computed from a per-image anomaly score (e.g. max or mean of the
  pixel-wise reconstruction error map).
- **Secondary/qualitative**: reconstruction + error heatmap grids per test
  image (good vs. defective side by side) — required visual artifact for
  write-up and interview demo.
- **Optional if time permits**: pixel-level localization metric (e.g. PRO or
  pixel AUC-ROC) using MVTec's provided ground-truth defect masks.

---

## 9. Experiment Tracking (W&B — minimal scope)

- `wandb.init(project="defectlens", config={...})` at the start of every
  run (baseline and every ablation), logging: category, loss type, encoder,
  fine-tune mode, latent_channels, lr, batch size, seed.
- `wandb.log({"epoch": e, "train_loss": ..., "val_loss": ..., "val_auc": ...})`
  per epoch.
- Log a small grid of `wandb.Image(...)` reconstructions every few epochs for
  qualitative tracking.
- **Deliberately not used**: W&B sweeps, artifact versioning, model registry —
  out of scope, adds complexity without portfolio value at this stage.
- **Why W&B despite being new to it**: it replaces manual run-tracking
  (renaming checkpoints, scrolling logs) with ~4 lines of code, and directly
  produces the ablation comparison tables Section 7 requires. It is also a
  documented gap across the other two resume projects.

---

## 10. Reproducibility

- Fixed seed (`torch.manual_seed`, `numpy.random.seed`, `random.seed`) set
  once at the top of `train.py`, same seed across ablation variants unless
  the ablation is explicitly testing seed sensitivity (it isn't, here).
- All hyperparameters per run come from `config/config.yaml`, never
  hardcoded inline — mirrors the documented-config principle from prior
  projects.

---

## 11. Build Order

Follow this order exactly. Each step has a clear verify condition. Do not
proceed to the next step until the current step is verified. **All steps
target `screw` first; `pill` is added only at Step 12.**

```
Step 1  — Environment setup (Colab + Drive)
          Mount Google Drive in the Colab notebook. Create the repo structure
          under a Drive-backed path so checkpoints survive disconnects.
          Verify: write a test file to the mounted Drive path, confirm it
          persists after runtime restart.

Step 2  — Download MVTec AD (screw category only)
          Download and extract the screw category into data/raw/screw/.
          Verify: train/good, test/good, test/<defect_types>/ subfolders exist
          with expected image counts.

Step 3  — Dataset class: src/data/dataset.py
          Implement MVTecDataset returning (x_encoder, x_target, label,
          mask_path_or_none) per sample, per Section 5.4's dual normalization.
          Verify: load one batch, assert x_encoder is ImageNet-normalized
          (mean~0) and x_target is in [0,1] (min>=0, max<=1).

Step 4  — Encoder wrapper: src/models/encoder.py
          Implement truncated ResNet18 (through layer3), pretrained=True,
          all params frozen by default.
          Verify: forward a test batch, assert output shape ~ [B, 256, 16, 16].

Step 5  — Bottleneck + Decoder: src/models/bottleneck.py, decoder.py
          Implement per Section 5.2/5.3.
          Verify: forward encoder output through bottleneck + decoder,
          assert final output shape == input image shape [B, 3, 256, 256].

Step 6  — Full autoencoder: src/models/autoencoder.py
          Wire encoder + bottleneck + decoder into one nn.Module.
          Verify: single forward pass end-to-end on one batch, no shape errors.

Step 7  — Naive loss: src/losses/mse_loss.py
          Implement plain MSE reconstruction loss.
          Verify: compute loss on one batch, assert scalar, non-negative.

Step 8  — Custom training loop: src/training/train.py
          Manual loop: forward, loss, loss.backward(), optimizer.step(),
          optimizer.zero_grad(), AMP via torch.cuda.amp.autocast + GradScaler,
          LR scheduling, checkpoint on best val loss. No Trainer.fit().
          Verify: train 2 epochs on screw with MSE loss, confirm loss
          decreases and a checkpoint is saved to Drive.

Step 9  — W&B integration
          Add wandb.init/log calls per Section 9. Keep minimal.
          Verify: one training run appears in the W&B dashboard with
          config + loss curve.

Step 10 — Evaluation metrics: src/eval/metrics.py
          Implement image-level AUC-ROC from per-image reconstruction error.
          Verify: run on screw test set (good + defective), assert AUC-ROC
          is a float in [0,1] and meaningfully above 0.5.

Step 11 — Visualization: src/eval/visualize.py
          Generate reconstruction + error heatmap grids (good vs defective).
          Verify: produces a saved figure under results/figures/ with
          recognizable structure (not pure noise).

          --- Screw baseline (plain MSE, frozen ResNet18) is now complete. ---

Step 12 — Add pill category
          Repeat Steps 2–3 dataset setup for pill. Confirm full pipeline
          (train + eval + viz) runs on pill without code changes beyond
          category config.
          Verify: pill baseline AUC-ROC computed and sane (above 0.5).

Step 13 — Composite loss: src/losses/ssim_composite_loss.py
          Implement alpha*L1 + beta*(1-SSIM). Use a standard SSIM
          implementation (e.g. torchmetrics or pytorch-msssim) rather than
          writing SSIM from scratch.
          Verify: compute loss on one batch, assert scalar, and that a
          perfect reconstruction yields near-zero loss.

Step 14 — Ablation 1: loss comparison
          Train screw (and pill) with MSE vs. composite loss, same all
          other settings. Log both to W&B, produce comparison table.
          Verify: results/ablation_tables/loss_ablation.csv exists with
          AUC-ROC for both variants, both categories.

Step 15 — Fine-tuning utilities: src/training/freeze_utils.py
          Implement freeze_encoder(model) and unfreeze_layers(model,
          layer_names) per Section 7.2's reasoning (layer3 [+layer2]
          unfrozen, conv1/layer1 stay frozen).
          Verify: assert requires_grad flags are correct after each call.

Step 16 — Ablation 2: fine-tuning strategy
          Train frozen vs. partial-unfreeze variant (best loss config from
          Step 14), same categories.
          Verify: comparison table produced with AUC-ROC + training stability
          notes (e.g. did unfrozen version overfit/diverge).

Step 17 — Ablation 3: latent dimension
          Sweep latent_channels in {256, 128, 64}, best config from Steps
          14–16.
          Verify: comparison table with AUC-ROC and qualitative reconstruction
          quality per latent size.

Step 18 — Ablation 4: backbone comparison
          Add EfficientNet-B0 encoder wrapper, run best config from Steps
          14–17 with both ResNet18 and EfficientNet-B0.
          Verify: comparison table with AUC-ROC vs. inference latency
          (measure via simple timed forward-pass loop) for both backbones.

Step 19 — Final write-up assets
          Architecture diagram, all four ablation tables, final metrics
          table (per category), reconstruction/heatmap figures.
          Verify: all assets present under results/, referenced in README.

Step 20 — (Optional, only if time remains) Cable as stretch category
          Repeat Steps 2–3, 10–11 for cable using the best locked config.
          Document where/why performance differs (structural defects being
          reconstructed too well) as a discussion point, not a fixed result.
```

---

## 12. Key Decisions and Constraints

### 12.1 Two categories, not one or four
One category risks looking thin for a portfolio piece; four (the original
candidate list of transistor/screw/cable/pill) risks debugging time spread
across categories with different failure modes before any baseline exists.
Screw + pill are both well-behaved for reconstruction-based methods, giving a
genuine cross-category result at controlled risk.

### 12.2 ResNet18 as primary encoder, EfficientNet-B0 ablation-only
ResNet18 is smaller, faster on a T4, and easier to explain block-by-block in
an interview. EfficientNet-B0 is introduced solely as the backbone-comparison
ablation (Section 7.4) — never the primary model.

### 12.3 Frozen encoder first, partial unfreeze later
Early conv layers capture generic low-level features (edges, textures) that
transfer well without retraining; only `layer3` (and optionally `layer2`) are
unfrozen for fine-tuning, since these are more task/domain-specific. This is
tested explicitly as Ablation 2, not assumed.

### 12.4 MSE first, composite SSIM loss second
Plain MSE is the naive baseline specifically so the MSE→composite comparison
is a real, logged ablation (Section 7.1), not just an assumed-better final
choice. Plain MSE alone is documented as insufficient for the final model due
to reconstruction blur (Section 6.1).

### 12.5 W&B kept, scope minimized
Considered dropping experiment tracking due to being new to PyTorch, but kept
it: the logging surface is ~4 lines of code, it replaces more error-prone
manual run-tracking, and it directly produces the required ablation
comparison tables. Advanced W&B features (sweeps, artifact registry) are
deliberately not used to avoid unnecessary scope.

### 12.6 Checkpointing to Google Drive, not local Colab disk
Colab free-tier sessions disconnect on idle/time limits; local Colab disk is
ephemeral and lost on disconnect. All checkpoints and W&B config must persist
to a mounted Drive path from Step 1 onward — this is infrastructure, decided
up front, not discovered mid-project after losing a training run.

### 12.7 Self-supervised pretraining and diffusion models excluded
Both have high "wow factor" and depth signal, but are too compute- and
time-intensive to execute rigorously within a 1-month window alongside
placement prep. A shallow, undebugged ambitious approach is worse than a
well-executed simpler one. Explicitly deferred, not planned for this
iteration.

---

## 13. Environment / Config Variables

All variables live in `config/config.yaml`. Never hardcode these inline.

```yaml
category: "screw"                # "screw" | "pill" | "cable" (stretch only)
image_size: 256

encoder:
  backbone: "resnet18"           # "resnet18" | "efficientnet_b0" (ablation only)
  pretrained: true
  truncate_at: "layer3"
  frozen: true                   # false for fine-tuning ablation

bottleneck:
  latent_channels: 256           # 256 | 128 | 64 for latent-dim ablation

loss:
  type: "composite"              # "mse" | "composite"
  alpha: 1.0
  beta: 1.0

training:
  batch_size: 16
  lr: 1e-3
  epochs: 50
  amp: true
  seed: 42
  checkpoint_dir: "/content/drive/MyDrive/defectlens/checkpoints"

wandb:
  project: "defectlens"
  mode: "online"                 # scalars + config + sample images only
```

---

## 14. .gitignore

```
# Data (large, re-downloadable)
data/raw/
data/splits/

# Checkpoints (large, stored on Drive not git)
results/checkpoints/

# Python
__pycache__/
*.pyc
.venv/
venv/

# Colab / Drive artifacts
*.ipynb_checkpoints/

# W&B local cache
wandb/

# IDE
.vscode/
.idea/
*.DS_Store
```
