"""Retrieval contract and batching regression tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_ROOT = ROOT / "retrieval"
EXTRA_ROOT = ROOT / "extra"
for candidate in (RETRIEVAL_ROOT, EXTRA_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from pytorch_export import PhysicalStateExportModule, _normalize_transition_days_with_stats
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


class _TinyNormalizedCoreModel(torch.nn.Module):
    """Small normalized-space core used to validate the physical export wrapper."""

    def forward(self, state0: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        return (
            state0[:, :1]
            + conditioning[:, :1].unsqueeze(-1).unsqueeze(-1)
            + conditioning[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        )


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
        "shape": {"C": 1, "H": 2, "W": 2},
        "state_fields": ["Phi"],
        "param_names": ["a_m"],
        "conditioning_names": ["a_m", "log10_transition_days"],
        "sampling": {
            "saved_checkpoint_interval_days": 1.0,
            "live_transition_days_min": 0.1,
            "live_transition_days_max": 10.0,
            "live_transition_tolerance_fraction": 0.1,
        },
        "normalization": {
            "state": {
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
                "artifact_kind": "direct_jump_physical_state_transition",
                "export_format": "torchscript",
                "supported_devices": ["cpu", "gpu"],
                "runtime_hints": {
                    "optimize_on_load": True,
                    "prefer_channels_last": True,
                    "allow_tf32_on_gpu": True,
                    "transition_time_feature": "log10_transition_days",
                },
                "physical_io": {
                    "state0": True,
                    "params": True,
                    "transition_days": True,
                    "state1": True,
                },
                "shape": {"C": 1, "H": 2, "W": 2},
                "input": {
                    "state0": ["batch", 1, 2, 2],
                    "params": ["batch", 1],
                    "transition_days": ["batch"],
                    "fields": ["Phi"],
                },
                "output": {
                    "state1": ["batch", 1, 2, 2],
                    "fields": ["Phi"],
                },
                "param_names": ["a_m"],
                "conditioning_names": ["a_m", "log10_transition_days"],
                "solver": {"dt_seconds": 240.0},
                "sampling": {
                    "saved_checkpoint_interval_days": 1.0,
                    "live_transition_days_min": 0.1,
                    "live_transition_days_max": 10.0,
                    "live_transition_tolerance_fraction": 0.1,
                },
                "normalization": checkpoint["normalization"],
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


def test_physical_state_export_module_uses_physical_inputs() -> None:
    """The export wrapper should normalize physical inputs and return physical outputs."""
    normalization = {
        "state": {
            "field_names": ["Phi"],
            "field_transforms": {"Phi": "none"},
            "mean": [0.0],
            "std": [1.0],
            "zscore_eps": 1.0e-8,
            "log10_eps": 1.0e-6,
            "signed_log1p_scale": 1.0,
        },
        "params": {
            "param_names": ["a_m"],
            "mean": [1.0],
            "std": [2.0],
            "is_constant": [False],
            "zscore_eps": 1.0e-8,
        },
        "transition_time": {
            "param_names": ["log10_transition_days"],
            "mean": [1.0],
            "std": [2.0],
            "is_constant": [False],
            "zscore_eps": 1.0e-8,
        },
    }
    export_model = PhysicalStateExportModule(
        model=_TinyNormalizedCoreModel(),
        normalization=normalization,
        state_fields=["Phi"],
    ).eval()
    state0 = torch.full((2, 1, 2, 2), 3.0, dtype=torch.float32)
    params = torch.tensor([[3.0], [5.0]], dtype=torch.float32)
    transition_days = torch.tensor([10.0, 1000.0], dtype=torch.float32)

    pred = export_model(state0, params, transition_days)

    expected = torch.tensor(
        [
            [[[4.0, 4.0], [4.0, 4.0]]],
            [[[6.0, 6.0], [6.0, 6.0]]],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(pred, expected, atol=1.0e-6)


def test_inspect_surrogate_artifact_accepts_direct_jump_torch_contract(
    tmp_path: Path,
) -> None:
    """A fresh direct-jump export bundle should validate without a checkpoint."""
    export_path, _ = _write_direct_jump_artifact(tmp_path)

    contract = inspect_surrogate_artifact(
        export_path=export_path,
        checkpoint_path=None,
        model_days=10.0,
    )

    assert contract.direct_jump_torch_runtime_ready is True
    assert contract.checkpoint_path is None
    assert contract.has_export_metadata is True
    assert contract.supported_devices == ("cpu", "gpu")
    assert contract.transition_days_conditioned is True
    assert contract.physical_io_forward is True
    assert contract.transition_time_name == "log10_transition_days"
    assert contract.sampling_saved_checkpoint_interval_days == pytest.approx(1.0)
    assert contract.sampling_live_transition_days_min == pytest.approx(0.1)
    assert contract.sampling_live_transition_days_max == pytest.approx(10.0)
    assert contract.blockers == ()


def test_inspect_surrogate_artifact_requires_cpu_gpu_supported_devices(
    tmp_path: Path,
) -> None:
    """The export metadata should keep the runtime device surface explicit."""
    export_path, _ = _write_direct_jump_artifact(tmp_path)
    meta_path = tmp_path / "model_export.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["supported_devices"] = ["cpu", "gpu", "tpu"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    contract = inspect_surrogate_artifact(
        export_path=export_path,
        checkpoint_path=None,
        model_days=10.0,
    )

    assert contract.direct_jump_torch_runtime_ready is False
    assert any(
        "supported_devices" in blocker for blocker in contract.torch_runtime_blockers
    )


def test_inspect_surrogate_artifact_requires_physical_io_flags(tmp_path: Path) -> None:
    """The export metadata must advertise physical input/output tensors explicitly."""
    export_path, _ = _write_direct_jump_artifact(tmp_path)
    meta_path = tmp_path / "model_export.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["physical_io"]["transition_days"] = False
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    contract = inspect_surrogate_artifact(
        export_path=export_path,
        checkpoint_path=None,
        model_days=10.0,
    )

    assert contract.direct_jump_torch_runtime_ready is False
    assert any("physical-space" in blocker for blocker in contract.torch_runtime_blockers)


def test_torch_surrogate_runtime_batches_predictions(tmp_path: Path) -> None:
    """The retrieval runtime should batch and broadcast valid inputs cleanly."""
    export_path, _ = _write_direct_jump_artifact(tmp_path)
    runtime = TorchSurrogateRuntime(
        export_path=export_path,
        checkpoint_path=None,
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


def test_torch_surrogate_runtime_keeps_tensor_output_on_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tensor-returning predictions should not force eager host copies."""
    export_path, _ = _write_direct_jump_artifact(tmp_path)
    runtime = TorchSurrogateRuntime(
        export_path=export_path,
        checkpoint_path=None,
        runtime_config=SurrogateRuntimeConfig(
            device_mode="cpu",
            max_batch_size=2,
            pin_host_memory=False,
            prefer_channels_last=False,
            allow_tf32=False,
        ),
        model_days=10.0,
    )

    def _unexpected_cpu(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("predict(return_numpy=False) should not call Tensor.cpu()")

    monkeypatch.setattr(torch.Tensor, "cpu", _unexpected_cpu, raising=False)

    state0 = np.ones((2, 1, 2, 2), dtype=np.float32)
    params = np.array([0.5], dtype=np.float32)
    transition_days = np.array([1.0, 2.0], dtype=np.float32)
    pred = runtime.predict(
        state0=state0,
        params=params,
        transition_days=transition_days,
        return_numpy=False,
    )

    expected = (
        torch.from_numpy(state0)
        + 0.5
        + torch.tensor(transition_days, dtype=torch.float32)[:, None, None, None]
    )
    assert isinstance(pred, torch.Tensor)
    assert pred.device.type == runtime.device.type
    assert torch.allclose(pred, expected, atol=1.0e-6)


def test_torch_surrogate_runtime_rejects_auto_device_mode(tmp_path: Path) -> None:
    """Runtime device selection should stay explicit for now."""
    export_path, checkpoint_path = _write_direct_jump_artifact(tmp_path)
    with pytest.raises(ValueError, match="Unsupported device_mode"):
        TorchSurrogateRuntime(
            export_path=export_path,
            checkpoint_path=checkpoint_path,
            runtime_config=SurrogateRuntimeConfig(device_mode="auto"),
            model_days=10.0,
        )
