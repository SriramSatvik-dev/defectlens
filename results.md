# DefectLens — Results

## Screw

| Config | max AUC | mean AUC | Notes |
|---|---|---|---|
| Baseline: MSE, frozen ResNet18 | 0.8436 | 0.0141 | Mean near-inverted |
| Composite loss (L1+SSIM), frozen ResNet18, lr=1e-4 | 0.8885 | 1.0000 | Verified clean separation |
| + Fine-tuned layer3 | 0.8727 | 1.0000 | max slightly down; `manipulated_front` weakest |
| Backbone: EfficientNet-B0, MSE, frozen | 0.8286 | 0.0054 | Near-replica of ResNet18 baseline |
| All combined (composite + finetune + EfficientNet) | 0.8000 | 1.0000 | max degrades further; `manipulated_front` → 0.58 |

**Best overall for screw:** Composite loss + fine-tuned ResNet18 (mean=1.0000, max=0.8727) — best balance of headline metric and per-defect-type robustness.

## Pill

| Config | max AUC | mean AUC | Notes |
|---|---|---|---|
| Baseline: MSE, frozen ResNet18 | 0.6947 | 0.8826 | Best pill config overall |
| Composite loss (L1+SSIM), frozen ResNet18, lr=1e-4 | 0.4523 | 0.4856 | Both near-random |
| + Fine-tuned layer3 (MSE, not composite) | 0.6269 | 0.6254 | `contamination` specifically degraded |
| Backbone: EfficientNet-B0, MSE, frozen | 0.6970 | 0.8034 | `color` hit hardest |
| All combined (composite + finetune + EfficientNet) | 0.4924 | 0.7697 | max worst-of-session; mean partially recovers |

**Best overall for pill:** Baseline (frozen ResNet18, plain MSE, mean aggregation) — every intervention tested underperformed it.

## Per-defect-type AUC-ROC (best config per category)

### Screw — composite loss + fine-tuned ResNet18

| Defect type | max | mean |
|---|---|---|
| manipulated_front | 0.7071 | 1.0000 |
| scratch_neck | 0.8748 | 1.0000 |
| scratch_head | 0.9343 | 1.0000 |
| thread_top | 0.9371 | 1.0000 |
| thread_side | 0.9614 | 1.0000 |

### Pill — baseline (MSE, frozen ResNet18)

| Defect type | max | mean |
|---|---|---|
| contamination | 0.5227 | 0.9242 |
| scratch | 0.5364 | 0.8477 |
| faulty_imprint | 0.6278 | 0.8636 |
| crack | 0.7190 | 0.7893 |
| color | 0.7684 | 0.8658 |
| combined | 0.8734 | 0.9968 |
| pill_type | 1.0000 | 1.0000 |
