"""
src/training/freeze_utils.py

Encoder freeze/partial-unfreeze logic, pulled out of train.py so it's
reusable and independently testable -- this is exactly the mechanism
Ablation 2 (fine-tuning) toggles between runs.
"""

import torch.nn as nn


def freeze_all(module: nn.Module):
    """Freeze every parameter in the given module (e.g. the whole encoder)."""
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_all(module: nn.Module):
    """Unfreeze every parameter in the given module."""
    for p in module.parameters():
        p.requires_grad = True


def partial_unfreeze_resnet18(encoder, unfreeze_layer3: bool = True, unfreeze_layer2: bool = False):
    """Ablation 2: unfreeze only later layers of a ResNet18Encoder, keep
    early layers (stem, layer1) frozen to avoid catastrophic forgetting
    of generic low-level features. Expects encoder to expose .stem,
    .layer1, .layer2, .layer3 (as ResNet18Encoder in src/models/encoder.py does).
    """
    freeze_all(encoder.stem)
    freeze_all(encoder.layer1)

    for p in encoder.layer2.parameters():
        p.requires_grad = unfreeze_layer2
    for p in encoder.layer3.parameters():
        p.requires_grad = unfreeze_layer3


def count_trainable_params(module: nn.Module) -> int:
    """Quick sanity-check helper: how many parameters currently have
    requires_grad=True in this module. Useful to print/log at the start
    of a run to visually confirm the freeze config actually took effect."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def apply_freeze_config(encoder, encoder_backbone: str, freeze_encoder: bool):
    """Single entrypoint train.py calls -- decides which freeze strategy
    to apply based on config, so train.py doesn't need if/else branches
    on backbone type itself."""
    if freeze_encoder:
        freeze_all(encoder)
    else:
        if encoder_backbone == "resnet18":
            # Partial unfreeze, not full -- see module docstring.
            partial_unfreeze_resnet18(encoder, unfreeze_layer3=True, unfreeze_layer2=False)
        else:
            # EfficientNet-B0 ablation path: full unfreeze for now, since
            # it's only used for the backbone-comparison ablation, not
            # combined with fine-tuning in this project's scope.
            unfreeze_all(encoder)


if __name__ == "__main__":
    # Verify condition: freeze/unfreeze actually changes requires_grad,
    # partial unfreeze leaves early layers frozen and later layers trainable.
    import torch
    from src.models.encoder import ResNet18Encoder

    encoder = ResNet18Encoder(freeze=True)
    print("After freeze=True init:", count_trainable_params(encoder), "trainable params (expect 0)")

    partial_unfreeze_resnet18(encoder, unfreeze_layer3=True, unfreeze_layer2=False)
    stem_trainable = count_trainable_params(encoder.stem)
    layer1_trainable = count_trainable_params(encoder.layer1)
    layer2_trainable = count_trainable_params(encoder.layer2)
    layer3_trainable = count_trainable_params(encoder.layer3)
    print(f"After partial unfreeze -- stem: {stem_trainable} (expect 0), "
          f"layer1: {layer1_trainable} (expect 0), "
          f"layer2: {layer2_trainable} (expect 0), "
          f"layer3: {layer3_trainable} (expect >0)")

    assert stem_trainable == 0 and layer1_trainable == 0 and layer2_trainable == 0
    assert layer3_trainable > 0
    print("Partial unfreeze verified: only layer3 is trainable, as configured.")