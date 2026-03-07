"""Retrieval contract and batching regression tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_ROOT = ROOT / "retrieval"
EXTRA_ROOT = ROOT / "extra"
for candidate in (RETRIEVAL_ROOT, EXTRA_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from pytorch_export import _normalize_transition_days_with_stats
from surrogate_backend import (
    SurrogateRuntimeConfig,
    TorchSurrogateRuntime,
    inspect_surrogate_artifact,
)


class _TinyDirectJumpModule(torch.nn.Module):
    """Small scripted surrogate used to validate retrieval batching."""

    def forward(
        self,
        state0: torch.Tensor,
        params: torch.Tensor,
        transition_days: torch.Tensor,
    ) -> torch.Tensor:
        scale = params[:, :1].unsqueeze(-1).unsqueeze(-1)
        return state0 + scale + transition_days[:, None, None, None]


def _write_direct_jump_artifact(tmp_path: Path) -> tuple[Path, Path]:
    """Write one minimal direct-jump export/checkpoint pair."""
    export_path = tmp_path / "model_export.torchscript.pt"
    checkpoint_path = tmp_path / "best.pt"
    meta_path = tmp_path / "model_export.meta.json"

    example_state = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    example_params = torch.tensor([[0.5]], dtype=torch.float32)
    example_transition_days = torch.tensor([10.0], dtype=torch.float32)
    scripted = torch.jit.trace(
        _TinyDirectJumpModule(),
        (example_state, example_params, example_transition_days),
        strict=True,
        check_trace=False,
    )
    scripted.save(str(export_path))

    checkpoint = {
        "solver": {"dt_seconds": 240.0},
        "shape": {"input_C": 1, "target_C": 1, "H": 2, "W": 2},
        "input_fields": ["Phi"],
        "target_fields": ["Phi"],
        "param_names": ["a_m"],
        "conditioning_names": ["a_m", "log10_transition_days"],
        "sampling": {
            "transition_jump_days_min": 0.1,
            "transition_jump_days_max": 10.0,
        },
        "normalization": {
            "input_state": {
                "field_names": ["Phi"],
                "field_transforms": {"Phi": "signed_log1p"},
                "mean": [0.0],
                "std": [1.0],
                "zscore_eps": 1.0e-8,
                "log10_eps": 1.0e-6,
                "signed_log1p_scale": 1.0,
            },
            "target_state": {
                "field_names": ["Phi"],
                "field_transforms": {"Phi": "signed_log1p"},
                "mean": [0.0],
                "std": [1.0],
                "zscore_eps": 1.0e-8,
                "log10_eps": 1.0e-6,
                "signed_log1p_scale": 1.0,
            },
            "params": {
                "param_names": ["a_m"],
                "mean": [0.5],
                "std": [1.0],
                "is_constant": [False],
                "zscore_eps": 1.0e-8,
            },
            "transition_time": {
                "param_names": ["log10_transition_days"],
                "mean": [1.0],
                "std": [1.0],
                "is_constant": [False],
                "zscore_eps": 1.0e-8,
            },
        },
    }
    torch.save(checkpoint, checkpoint_path)
    meta_path.write_text(
        json.dumps(
            {
                "input": {
                    "state0": ["batch", 1, 2, 2],
                    "params": ["batch", 1],
                    "transition_days": ["batch"],
                }
            }
        ),
        encoding="utf-8",
    )
    return export_path, checkpoint_path


def test_export_transition_days_normalization_uses_log10() -> None:
    """Export conditioning should log-encode physical transition durations."""
    transition_days = torch.tensor([0.1, 10.0], dtype=torch.float32)
    normalized = _normalize_transition_days_with_stats(
        transition_days,
        mean=torch.tensor([0.0], dtype=torch.float32),
        std=torch.tensor([1.0], dtype=torch.float32),
        is_constant=torch.tensor([False]),
        zscore_eps=1.0e-8,
    )
    expected = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    assert torch.allclose(normalized, expected, atol=1.0e-6)


def test_inspect_surrogate_artifact_accepts_direct_jump_torch_contract(
    tmp_path: Path,
) -> None:
    """A fresh direct-jump Torch export should pass Torch-runtime validation."""
    export_path, checkpoint_path = _write_direct_jump_artifact(tmp_path)

    contract = inspect_surrogate_artifact(
        export_path=export_path,
        checkpoint_path=checkpoint_path,
        model_days=10.0,
    )

    assert contract.direct_jump_torch_runtime_ready is True
    assert contract.transition_days_conditioned is True
    assert contract.transition_time_name == "log10_transition_days"
    assert contract.gpu_native_jax_retrieval_ready is False


def test_torch_surrogate_runtime_batches_predictions(tmp_path: Path) -> None:
    """The retrieval runtime should batch and broadcast valid inputs cleanly."""
    export_path, checkpoint_path = _write_direct_jump_artifact(tmp_path)
    runtime = TorchSurrogateRuntime(
        export_path=export_path,
        checkpoint_path=checkpoint_path,
        runtime_config=SurrogateRuntimeConfig(
            device_mode="cpu",
            max_batch_size=2,
            pin_host_memory=False,
            prefer_channels_last=False,
            allow_tf32=False,
        ),
        model_days=10.0,
    )

    state0 = np.ones((5, 1, 2, 2), dtype=np.float32)
    params = np.array([0.5], dtype=np.float32)
    transition_days = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    pred = runtime.predict(
        state0=state0,
        params=params,
        transition_days=transition_days,
        return_numpy=True,
    )

    expected = state0 + 0.5 + transition_days[:, None, None, None]
    assert pred.shape == state0.shape
    assert np.allclose(pred, expected, atol=1.0e-6)
