"""Scheduler helper regression tests."""

from __future__ import annotations

import pytest
import torch

from training import (
    _early_stopping_patience,
    _linear_warmup_lr,
    _loss_improved,
    _reduce_plateau_learning_rate,
    _update_loss_tracking,
)


def test_linear_warmup_lr_reaches_base_lr_at_warmup_boundary() -> None:
    """Linear warmup should ramp monotonically up to the configured base LR."""
    assert _linear_warmup_lr(epoch=1, base_lr=1.0e-3, warmup_epochs=5) == pytest.approx(2.0e-4)
    assert _linear_warmup_lr(epoch=3, base_lr=1.0e-3, warmup_epochs=5) == pytest.approx(6.0e-4)
    assert _linear_warmup_lr(epoch=5, base_lr=1.0e-3, warmup_epochs=5) == pytest.approx(1.0e-3)
    assert _linear_warmup_lr(epoch=8, base_lr=1.0e-3, warmup_epochs=5) == pytest.approx(1.0e-3)


def test_loss_improved_uses_absolute_min_delta() -> None:
    """Small changes below the epsilon threshold should not reset patience."""
    assert _loss_improved(current=0.9, best=1.0, min_delta=1.0e-10) is True
    assert _loss_improved(current=0.99999999995, best=1.0, min_delta=1.0e-10) is False


def test_plateau_lr_counts_warmup_bad_epochs() -> None:
    """Plateau decay should honor validation stalls that begin during warmup."""
    param = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([param], lr=1.0e-3)
    best = float("inf")
    bad_epochs = 0
    warmup_epochs = 3
    patience = 2

    for epoch, val_loss in enumerate((1.0, 0.5, 0.6, 0.7), start=1):
        best, bad_epochs, _ = _update_loss_tracking(
            current=val_loss,
            best=best,
            bad_epochs=bad_epochs,
            min_delta=1.0e-10,
        )
        if epoch >= warmup_epochs and bad_epochs >= patience:
            if _reduce_plateau_learning_rate(
                optimizer,
                factor=0.5,
                min_lr=1.0e-5,
                eps=1.0e-10,
            ):
                bad_epochs = 0

    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-4)


def test_early_stopping_patience_allows_multiple_lr_reductions() -> None:
    """Early stopping should remain more patient than a single plateau window."""
    assert _early_stopping_patience(scheduler_patience=10, warmup_epochs=5) == 30
