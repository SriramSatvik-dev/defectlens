"""
src/training/scheduler.py

Learning rate scheduler setup, pulled out of train.py so scheduler choice
is a swappable/testable unit on its own.

Default: ReduceLROnPlateau, watching val AUC-ROC (mode="max" -- higher is
better) rather than val loss -- consistent with checkpointing on val AUC
rather than val loss (same reasoning as before: loss can keep improving
while AUC-ROC plateaus or degrades, so the scheduler should react to the
metric that actually matters, not the one that's easiest to differentiate).

CosineAnnealingLR is offered as an alternative for experimentation, since
it doesn't depend on watching a metric at all (pure schedule by epoch),
which is sometimes more stable for short ablation runs where plateau
detection barely has time to trigger.
"""

import torch


def build_scheduler(optimizer, scheduler_type: str = "plateau", epochs: int = 50,
                     patience: int = 5, factor: float = 0.5):
    if scheduler_type == "plateau":
        # mode="max" because we're watching val AUC-ROC, not loss.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


def step_scheduler(scheduler, scheduler_type: str, val_auc: float):
    """Single entrypoint train.py calls after each epoch -- handles the
    fact that ReduceLROnPlateau needs the watched metric passed to .step(),
    while CosineAnnealingLR takes no argument at all."""
    if scheduler is None:
        return
    if scheduler_type == "plateau":
        scheduler.step(val_auc)
    elif scheduler_type == "cosine":
        scheduler.step()


if __name__ == "__main__":
    # Verify condition: scheduler reduces LR after enough non-improving
    # epochs, on a synthetic flat val_auc sequence.
    model_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([model_param], lr=1e-3)
    scheduler = build_scheduler(optimizer, scheduler_type="plateau", patience=2, factor=0.5)

    initial_lr = optimizer.param_groups[0]["lr"]
    print("Initial LR:", initial_lr)

    # Simulate a flat val_auc (no improvement) for several epochs.
    flat_auc = 0.75
    for epoch in range(6):
        step_scheduler(scheduler, "plateau", flat_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}: val_auc={flat_auc}, lr={current_lr}")

    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr < initial_lr, "LR should have dropped after patience epochs of no improvement"
    print(f"Verified: LR dropped from {initial_lr} to {final_lr} after plateau.")