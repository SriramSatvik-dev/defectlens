# DefectLens — Decisions & Findings

Project-specific design decisions and empirical findings from building DefectLens. (General, transferable technical concepts belong in your own separate notes, not here.)

## Design Decisions

**Unsupervised reconstruction over supervised classification.** Chosen because: (1) real manufacturing defects are rare and a supervised classifier only recognizes defect types it was shown; (2) MVTec's defect counts are small (dozens per subtype) — too little to train a generalizing classifier; (3) this approach specifically demonstrates training-loop and loss-design skills that FinQuery/PhishGuard don't cover.

**Val set: stratified good + defective, not good-only.** Training only ever sees good images, but the metric that matters — separation between good and defective reconstruction error — can't be measured without defective examples in val too. ~15% of `test/` stratified per defect type into val; defect types under 15 images (pill's `pill_type`, 9 images) excluded from stratification, sent entirely to final test.

**Checkpoint on best val AUC-ROC, not best val loss.** Directly justified by the pill baseline run: val_auc peaked at epoch 10 (0.81) then declined to 0.52 by epoch 50 while train_loss kept improving the entire time. Saving on loss alone would have kept a materially worse model.

**Dual normalization.** Encoder input uses ImageNet mean/std (`[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`), matching what the pretrained ResNet18/EfficientNet-B0 weights expect. Reconstruction target stays in plain [0,1], matching the decoder's Sigmoid output. Two forms of the same image, for two different purposes.

**Frozen encoder by default, partial unfreeze for Ablation 2.** Stem+layer1 always frozen even when fine-tuning; only layer3 unfrozen, to avoid catastrophic forgetting of generic low-level features while still allowing task-specific adaptation.

**Learning rate re-tuned per loss function, not fixed.** MSE's lr=1e-3 caused a sharp training collapse when reused for composite loss (val_auc 0.81→0.13 within one epoch on screw); lr=1e-4 trained smoothly to a higher peak (0.87). Confirmed via direct rerun comparison that this was optimizer instability from SSIM's differently-scaled gradients, not identity-mapping.

## Findings

**1. MSE causes near-total score inversion on screw under mean aggregation.** Blurry MSE reconstructions make screw's structural defects (locally simpler/smoother than intact fine thread texture) easier to reconstruct than the surrounding normal texture — so defective images can average *lower* error than good ones. Confirmed via direct per-image inspection (good images mean=0.105, defective mean=0.166 under a config where the direction was reversed) — a real, systematic effect, not noise.

**2. No universal best aggregation method — depends on defect geometry, confirmed empirically per category.** Screw's localized, structural defects favor max (checks only the single worst point, unaffected by the rest of the image). Pill's broader defects, especially contamination, favor mean (a diffuse anomaly's cumulative signal is diluted rather than found by max). General pattern observed but not proven universal: max pairs better with MSE loss, mean pairs better with composite loss.

**3. Composite loss changes error geometry — why it flips mean aggregation from harmful to near-perfect on screw.** MSE's blur produces narrow error spikes only at defects (favors max). Composite loss's sharper, structure-aware reconstruction produces broad, elevated error across the whole defective region (favors mean too). Verified via score distributions: tight, non-overlapping clusters (good: 0.097–0.116, defective: 0.135–0.198) — genuine separation, not a fragile artifact, despite the suspicious-looking 1.0000 AUC.

**4. Composite loss is not a universal improvement — it is defect-character-dependent.** Helped screw substantially (max 0.84→0.89, mean 0.01→1.00). Actively hurt pill below its own MSE baseline (max 0.69→0.45, mean 0.88→0.49) across every pill configuration tested (composite alone, +finetune, +finetune+EfficientNet) — not a one-off. Mechanism: SSIM's structural-similarity bias competes against the color/intensity fidelity pill's defects actually depend on.

**5. Fine-tuning and reduced backbone capacity degrade pill via the same underlying mechanism.** Three independent interventions (composite loss, fine-tuning layer3, narrower EfficientNet-B0 bottleneck: 112 vs 256 channels) each hurt pill by eroding the same resource — broad, ImageNet-pretrained color/intensity sensitivity. Fine-tuning: `contamination` AUC dropped 0.92→0.58. Narrower backbone: `color` AUC dropped 0.87→0.67, while structurally-obvious types (`combined`, `pill_type`) were largely unaffected. Stacking all three did not purely compound — max hit its worst point of the whole project (0.49), but mean partially recovered (0.77) relative to the two weakest single-lever runs, showing the degradation mechanisms overlap rather than add linearly.

**6. Backbone choice matters far less than loss function or defect-signal alignment — except where channel capacity is the binding constraint.** EfficientNet-B0 nearly replicated ResNet18's screw baseline exactly (max 0.83 vs 0.84, same near-total mean inversion) despite very different architecture and less than half the channels. The one place backbone capacity clearly mattered was pill's `color` defect type specifically, confirming the exception is narrow.

**7. Training curve stability is not a proxy for model quality.** Combined-ablation runs converged faster and more smoothly than single-lever ablations. For pill this reflected a uniformly mediocre plateau (~0.55 val_auc). For screw it reflected a good aggregate result that concealed a real regression in the hardest defect type (`manipulated_front`, max AUC 0.71→0.58) — visible only in the per-defect-type test breakdown, not in the val curve.

**8. Val-set AUC-ROC (n≈24–25) is too noisy to make final decisions from.** Every "best" result in this project was re-verified on the larger (~136–142 image) test set before being trusted. Two results that looked suspicious on first glance (near-zero AUC, perfect 1.0000 AUC) were investigated via direct per-image score inspection and confirmed genuine rather than assumed to be bugs or accepted at face value.

## Scope Notes

- No deployment threshold layer was built — the system outputs continuous anomaly scores evaluated via AUC-ROC (ranking quality), not a good/defective classifier. Converting scores to decisions would require a threshold chosen from the real-world cost asymmetry between false positives and false negatives (e.g. Youden's J statistic, or a percentile-of-good-scores cutoff) — a business decision outside this project's scope.
- SSIM window size (11, library default) and `latent_channels` (256) were treated as explicit, stated decisions but not empirically ablated.
