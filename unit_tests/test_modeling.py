"""Modeling and loss regression tests."""

from __future__ import annotations

import pytest
import torch

from gcmulator.modeling import SphereLoss, StateConditionedTransitionModel


class _IdentityBase(torch.nn.Module):
    """Small SFNO-like stub for residual-path unit tests."""

    def __init__(self, *, embed_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.encoder = torch.nn.Identity()
        self.pos_embed = torch.nn.Identity()
        self.pos_drop = torch.nn.Identity()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(num_layers)])
        self.decoder = torch.nn.Identity()


def test_state_conditioned_transition_model_uses_fixed_big_skip() -> None:
    """Residual prediction should add the decoder output back 1:1 to the input state."""
    model = StateConditionedTransitionModel(
        base=_IdentityBase(embed_dim=3, num_layers=1),
        param_dim=2,
        input_state_chans=3,
        target_state_chans=3,
        nlat=2,
        nlon=2,
        residual_prediction=True,
    )
    state0 = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    params = torch.zeros((1, 2), dtype=torch.float32)

    pred = model(state0, params)

    assert torch.allclose(pred, 2.0 * state0)


def test_state_conditioned_transition_model_can_disable_big_skip() -> None:
    """Disabling residual prediction should return the decoder output directly."""
    model = StateConditionedTransitionModel(
        base=_IdentityBase(embed_dim=3, num_layers=1),
        param_dim=2,
        input_state_chans=3,
        target_state_chans=3,
        nlat=2,
        nlon=2,
        residual_prediction=False,
    )
    state0 = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    params = torch.zeros((1, 2), dtype=torch.float32)

    pred = model(state0, params)

    assert torch.allclose(pred, state0)


def test_sphere_loss_equal_weights_match_unweighted_baseline() -> None:
    """Explicit equal channel weights should preserve the original scalar loss."""
    pytest.importorskip("torch_harmonics.examples.losses")
    pred = torch.zeros((2, 3, 4, 8), dtype=torch.float32)
    target = torch.ones((2, 3, 4, 8), dtype=torch.float32)

    baseline = SphereLoss(nlat=4, nlon=8, grid="legendre-gauss")
    weighted = SphereLoss(
        nlat=4,
        nlon=8,
        grid="legendre-gauss",
        channel_weights=[1.0, 1.0, 1.0],
    )

    baseline_value = baseline(pred, target)
    weighted_value, per_channel = weighted.loss_with_channels(pred, target)

    assert torch.allclose(weighted_value, baseline_value)
    assert per_channel.shape == (3,)


def test_sphere_loss_applies_custom_channel_weights_in_field_order() -> None:
    """The aggregate loss should be the weighted mean of ordered per-channel losses."""
    pytest.importorskip("torch_harmonics.examples.losses")
    pred = torch.zeros((1, 3, 4, 8), dtype=torch.float32)
    target = torch.stack(
        [
            torch.full((4, 8), 1.0, dtype=torch.float32),
            torch.full((4, 8), 2.0, dtype=torch.float32),
            torch.full((4, 8), 3.0, dtype=torch.float32),
        ],
        dim=0,
    ).unsqueeze(0)
    channel_weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    loss = SphereLoss(
        nlat=4,
        nlon=8,
        grid="legendre-gauss",
        channel_weights=channel_weights.tolist(),
    )

    aggregate, per_channel = loss.loss_with_channels(pred, target)
    expected = torch.sum(per_channel * channel_weights) / torch.sum(channel_weights)

    assert per_channel.shape == (3,)
    assert per_channel[0] < per_channel[1] < per_channel[2]
    assert torch.allclose(aggregate, expected)
