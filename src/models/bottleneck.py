"""
bottleneck.py

1x1 conv bridging the encoder's native output channels to a tunable
latent_channels dimension. Exists so:
1. latent_channels can be tuned/ablated independently of which encoder
   backbone is used (ResNet18: 256 channels, EfficientNet-B0: 112 channels).
2. Swapping encoders (Ablation 3/4) doesn't require touching the decoder --
   the bottleneck absorbs the channel-count mismatch.
"""

import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, latent_channels, kernel_size=1)
        self.out_channels = latent_channels

    def forward(self, x):
        return self.conv(x)