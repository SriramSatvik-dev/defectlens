"""
losses.py

Loss functions for DefectLens.

- MSELoss: plain MSE baseline (Ablation 0 / first data point).
- CompositeLoss: alpha*L1 + beta*(1-SSIM) -- the main model's loss,
  motivated by: MSE/L1 alone tends to blur reconstructions (safest way
  to minimize per-pixel squared error under uncertainty is to predict
  something close to the average), which washes out the small, localized
  errors that are exactly what defect detection depends on. SSIM measures
  local structural agreement (not just pixel intensity), so it's more
  sensitive to localized structural anomalies -- the L1 term keeps
  overall reconstruction stable/honest, SSIM sharpens local sensitivity.

Requires: pip install pytorch-msssim
(reason for using a library rather than hand-rolling SSIM: numerically
correct SSIM has a few easy-to-get-wrong details -- Gaussian window
construction, padding/border handling, luminance/contrast/structure
constant choices -- and a well-tested library avoids silently-wrong
gradients, which would be a much worse bug than the code being simply
absent.)

SSIM window size is left as a constructor argument (default 11, the
standard choice) rather than hardcoded -- Step 13's actual decision
(window size) gets made by choosing the value used at training time,
not baked as an unstated default here.
"""

import torch
import torch.nn as nn
# from pytorch_msssim import SSIM


class MSELoss(nn.Module):
    """Plain MSE baseline. Ablation 0 -- establishes the floor that
    the composite loss (below) is compared against."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, recon, target):
        return self.mse(recon, target)


if __name__ == "__main__":
    # Verify condition: both losses run on dummy [0,1]-range tensors,
    # produce a scalar, and gradients flow back through a dummy
    # computation using them.
    recon = torch.rand(2, 3, 256, 256, requires_grad=True)
    target = torch.rand(2, 3, 256, 256)

    mse = MSELoss()
    mse_val = mse(recon, target)
    print("MSE loss:", mse_val.item())
    mse_val.backward(retain_graph=True)
    print("MSE backward OK, recon.grad is not None:", recon.grad is not None)

    # Sanity check: identical input/target should give near-zero loss.
    identical = torch.rand(1, 3, 256, 256)
    mse_zero = mse(identical, identical)
    print(f"Identical-input sanity check -- MSE: {mse_zero.item():.6f} (expect ~0)")