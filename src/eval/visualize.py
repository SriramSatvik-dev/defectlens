import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_map(x_target: torch.Tensor, recon: torch.Tensor, error_map: torch.Tensor, title: str = "Map Visualization"):
    """
    Plots the target, reconstruction, and error map side by side.
    x_target, recon: [3, H, W] tensors, in [0,1]
    error_map: [H, W] tensor (already channel-averaged, from metrics.py)
    """
    # Convert to numpy, move channels to last axis for RGB images
    target_np = x_target.detach().cpu().permute(1, 2, 0).numpy()
    recon_np = recon.detach().cpu().permute(1, 2, 0).numpy()
    error_np = error_map.detach().cpu().numpy()  # already [H, W], no permute needed

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(target_np)
    axs[0].set_title('Target')
    axs[0].axis('off')

    axs[1].imshow(recon_np)
    axs[1].set_title('Reconstruction')
    axs[1].axis('off')

    axs[2].imshow(error_np, cmap='hot')
    axs[2].set_title('Error Map')
    axs[2].axis('off')

    plt.suptitle(title)
    plt.show()