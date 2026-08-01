"""
autoencoder.py

Wires encoder + bottleneck + decoder into the full DefectLens autoencoder.

Default config: ResNet18Encoder (frozen), latent_channels=256.
Backbone swap (EfficientNetB0Encoder) and freeze/unfreeze are both
handled by passing a different encoder instance in -- this file doesn't
hardcode the choice, so Ablation 2 (fine-tuning) and Ablation 3/4
(backbone comparison) don't require editing this file at all.
"""

import torch
import torch.nn as nn

from .encoder import ResNet18Encoder, EfficientNetB0Encoder
from .bottleneck import Bottleneck
from .decoder import Decoder


class DefectLensAE(nn.Module):
    def __init__(self, encoder: nn.Module = None, latent_channels: int = 256):
        super().__init__()
        # Default to the primary ResNet18 encoder, frozen, if none given.
        self.encoder = encoder if encoder is not None else ResNet18Encoder(freeze=True)
        self.bottleneck = Bottleneck(self.encoder.out_channels, latent_channels)
        self.decoder = Decoder(in_channels=latent_channels)

    def forward(self, x_encoder):
        z = self.encoder(x_encoder)
        z = self.bottleneck(z)
        recon = self.decoder(z)
        return recon


if __name__ == "__main__":
    # Verify condition: shapes flow correctly end to end, encoder frozen
    # by default, decoder params require grad, output bounded in [0,1].
    model = DefectLensAE()
    dummy = torch.randn(2, 3, 256, 256)  # batch of 2, matches dataset.py output shape
    out = model(dummy)
    print("Input shape:", dummy.shape)
    print("Output shape:", out.shape)
    assert out.shape == dummy.shape, "Reconstruction shape must match input shape"

    encoder_trainable = sum(p.requires_grad for p in model.encoder.parameters())
    decoder_trainable = sum(p.requires_grad for p in model.decoder.parameters())
    print(f"Encoder params requiring grad: {encoder_trainable} (expect 0, frozen by default)")
    print(f"Decoder params requiring grad: {decoder_trainable} (expect >0, trained from scratch)")
    print("Output min/max:", out.min().item(), out.max().item(), "(expect within [0,1])")

    # Quick sanity check that the EfficientNet-B0 ablation path also wires
    # up correctly (channel mismatch absorbed by bottleneck).
    eff_model = DefectLensAE(encoder=EfficientNetB0Encoder(freeze=True))
    eff_out = eff_model(dummy)
    assert eff_out.shape == dummy.shape, "EfficientNet-B0 path must also reconstruct full size"
    print("EfficientNet-B0 ablation path output shape:", eff_out.shape, "(matches input, as expected)")