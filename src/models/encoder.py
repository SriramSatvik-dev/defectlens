"""
encoder.py

Encoder wrappers for DefectLens.

- ResNet18Encoder: primary encoder, truncated after layer3, frozen by
  default. Ablation 2 (fine-tuning) uses set_partial_unfreeze() on this.
- EfficientNetB0Encoder: reserved SOLELY for the backbone-comparison
  ablation (Ablation 3/4) -- not used in the main model. Kept separate
  and clearly labeled so it's never accidentally wired into the default
  pipeline.
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CustomConvEncoder(nn.Module):
    """From-scratch convolutional encoder, modeled on Bergmann et al. 2018
    (SSIM autoencoder). Trained jointly with the decoder, no pretraining.

    Input:  ImageNet-normalized [B, 3, 256, 256]
            (BatchNorm in the first block absorbs the normalization scale,
            so dataset.py doesn't need to change.)
    Output: spatial feature map [B, out_channels, 16, 16]

    Four stride-2 convs give /16 downsampling, matching the pretrained
    encoders, so bottleneck.py and decoder.py work unchanged.

    freeze defaults to False: a frozen randomly-initialized encoder is
    meaningless. Don't use it with --freeze_encoder (see freeze_utils edit).
    """

    def __init__(self, freeze: bool = False, out_channels: int = 32):
        super().__init__()

        def block(in_ch, out_ch, k, s):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            block(3, 32, 4, 2),     # 256 -> 128
            block(32, 32, 4, 2),    # 128 -> 64
            block(32, 32, 3, 1),    #  64 -> 64
            block(32, 64, 4, 2),    #  64 -> 32
            block(64, 64, 3, 1),    #  32 -> 32
            block(64, 128, 4, 2),   #  32 -> 16
            block(128, 64, 3, 1),   #  16 -> 16
            block(64, 32, 3, 1),    #  16 -> 16
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

        self.out_channels = out_channels
        self.set_frozen(freeze)

    def set_frozen(self, freeze: bool):
        for p in self.parameters():
            p.requires_grad = not freeze

    def forward(self, x):
        return self.net(x)

class ResNet18Encoder(nn.Module):
    """Pretrained ResNet18, truncated after layer3.
    Input: ImageNet-normalized [B, 3, 256, 256]
    Output: spatial feature map [B, 256, 16, 16]
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        base = resnet18(weights=weights)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        self.out_channels = 256
        self.set_frozen(freeze)

    def set_frozen(self, freeze: bool):
        """Freeze/unfreeze all encoder params."""
        for p in self.parameters():
            p.requires_grad = not freeze

    def set_partial_unfreeze(self, unfreeze_layer3: bool = True, unfreeze_layer2: bool = False):
        """Ablation 2: unfreeze only later layers, keep stem/layer1 frozen
        to avoid catastrophic forgetting of generic low-level features."""
        for p in self.stem.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False
        for p in self.layer2.parameters():
            p.requires_grad = unfreeze_layer2
        for p in self.layer3.parameters():
            p.requires_grad = unfreeze_layer3

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # [B, 256, 16, 16]


class EfficientNetB0Encoder(nn.Module):
    """Pretrained EfficientNet-B0, truncated to a comparable spatial
    downsampling factor (/16) as ResNet18Encoder above, for the
    backbone-comparison ablation ONLY. Not used in the default model.

    EfficientNet-B0's `features` is a sequential of 9 blocks (indices 0-8).
    Blocks 0-5 bring you to /16 spatial downsampling with 112 channels --
    close in spirit to ResNet18's layer3 cutoff. Exact channel count
    differs from ResNet18 (112 vs 256), which bottleneck.py's tunable
    1x1 conv absorbs.
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        base = efficientnet_b0(weights=weights)

        # Truncate at block index 5 (0-indexed) -> /16 downsampling, 112 channels.
        self.features = nn.Sequential(*list(base.features.children())[:6])
        self.out_channels = 112

        self.set_frozen(freeze)

    def set_frozen(self, freeze: bool):
        for p in self.parameters():
            p.requires_grad = not freeze

    def forward(self, x):
        return self.features(x)  # [B, 112, 16, 16]