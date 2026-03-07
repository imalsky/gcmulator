"""Scheduler helper regression tests."""

from __future__ import annotations

import pytest

from training import _early_stopping_patience, _linear_warmup_lr, _loss_improved


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


def test_early_stopping_patience_allows_multiple_lr_reductions() -> None:
    """Early stopping should remain more patient than a single plateau window."""
    assert _early_stopping_patience(scheduler_patience=10, warmup_epochs=5) == 30
