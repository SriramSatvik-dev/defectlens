"""
decoder.py

Custom ConvTranspose2d decoder. Takes the bottleneck's [B, latent_channels,
16, 16] feature map and upsamples back to [B, 3, 256, 256], ending in
Sigmoid so output range matches x_target's [0,1] range from dataset.py.

Each block roughly doubles spatial size while halving channels:
16->32->64->128->256, channels latent_channels->128->64->32->16->3.
"""

import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            up_block(in_channels, 128),  # 16 -> 32
            up_block(128, 64),           # 32 -> 64
            up_block(64, 32),            # 64 -> 128
            up_block(32, 16),            # 128 -> 256
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),                # output in [0,1], matches x_target
        )

    def forward(self, z):
        return self.net(z)