import torch
import torch.nn as nn
from pytorch_msssim import SSIM

class CompositeLoss(nn.Module):
    """alpha * L1 + beta * (1 - SSIM)

    alpha, beta: relative weighting between the two terms. Defaults
    (1.0, 1.0) are a starting point, not a locked decision -- worth
    treating the ratio as tunable if the composite loss's ablation
    result looks off in an unexpected direction (e.g. reconstructions
    look "SSIM-plausible" but pixel-wise noisy, or vice versa).

    win_size: SSIM window size. Default 11 (the library default, and
    the standard choice in the literature) -- explicit here so it's
    a stated decision, not a silent default. Revisit only if
    reconstructions/heatmaps look off (Step 13's actual empirical check).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, win_size: int = 11):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()
        # data_range=1.0 because x_target/recon are both in [0,1]
        # (Sigmoid output, matching dataset.py's x_target range).
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3, win_size=win_size)

    def forward(self, recon, target):
        l1_term = self.l1(recon, target)
        ssim_term = 1 - self.ssim(recon, target)
        loss = self.alpha * l1_term + self.beta * ssim_term
        return loss, {"l1": l1_term.item(), "ssim_term": ssim_term.item()}

if __name__ == "__main__":
    recon = torch.rand(2, 3, 256, 256, requires_grad=True)
    target = torch.rand(2, 3, 256, 256)

    composite = CompositeLoss()
    comp_val, components = composite(recon, target)
    print("Composite loss:", comp_val.item(), "| components:", components)
    comp_val.backward()
    print("Composite backward OK, recon.grad is not None:", recon.grad is not None)

    identical = torch.rand(1, 3, 256, 256)
    comp_zero, _ = composite(identical, identical)
    print(f"Identical-input sanity check -- Composite: {comp_zero.item():.6f} (expect ~0)")