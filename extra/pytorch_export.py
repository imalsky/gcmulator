"""Export utility to package trained model as physical-space TorchScript module."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict
import warnings

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import build_rollout_model, ensure_torch_harmonics_importable

# Global configuration.
MODEL_DIR = Path("models/model")
CHECKPOINT_NAME = "best.pt"
EXPORT_NAME = "model_export.torchscript.pt"
EXPORT_META_NAME = "model_export.meta.json"

DEVICE_MODE = "auto"

EXAMPLE_BATCH_SIZE = 1
STRICT_TRACE = True
SUPPRESS_KNOWN_EXPORT_WARNINGS = True

# Normalization constants baked into exported physical-space model behavior.
FIELD_MODE_NONE = 0
FIELD_MODE_LOG10 = 1
FIELD_MODE_SIGNED_LOG1P = 2
LOG10_INVERSE_CLIP_MIN = -30.0
LOG10_INVERSE_CLIP_MAX = 30.0
PARAM_NORM_CLIP_ABS = 1.0e6


def _dict_to_namespace(obj: Any) -> Any:
    """Recursively convert nested dict/list structures into namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(x) for x in obj]
    return obj


def _resolve_model_dir(path_value: Path) -> Path:
    """Resolve model directory from absolute, repo-relative, or cwd-relative path."""
    if path_value.is_absolute():
        return path_value.resolve()

    # Default behavior: treat relative paths as repo-root relative.
    project_candidate = (PROJECT_ROOT / path_value).resolve()
    if project_candidate.exists():
        return project_candidate

    # Fallback: allow cwd-relative custom paths if the user set one.
    cwd_candidate = path_value.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return project_candidate


def _resolve_device(mode: str) -> torch.device:
    """Resolve export device from utility configuration."""
    m = str(mode).lower()
    if m == "cpu":
        return torch.device("cpu")
    if m == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("DEVICE_MODE='gpu' but CUDA is unavailable")
    if m == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"Unsupported DEVICE_MODE={mode}. Use 'cpu', 'gpu', or 'auto'.")


def _setup_warning_filters() -> None:
    """Suppress known non-actionable tracing/export warnings."""
    if not SUPPRESS_KNOWN_EXPORT_WARNINGS:
        return
    warnings.filterwarnings(
        "ignore",
        message=r"Casting complex values to real discards the imaginary part.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Converting a tensor to a Python boolean might cause the trace to be incorrect.*",
        category=torch.jit.TracerWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Converting a tensor to a Python integer might cause the trace to be incorrect.*",
        category=torch.jit.TracerWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"An output with one or more elements was resized since it had shape \[\],.*",
        category=UserWarning,
    )


def _build_state_mode_codes(*, fields: list[str], field_transforms: Dict[str, Any]) -> list[int]:
    """Encode string transforms into small integer mode IDs."""
    modes: list[int] = []
    for name in fields:
        tr = str(field_transforms.get(name, "none"))
        if tr == "none":
            modes.append(FIELD_MODE_NONE)
        elif tr == "log10":
            modes.append(FIELD_MODE_LOG10)
        elif tr == "signed_log1p":
            modes.append(FIELD_MODE_SIGNED_LOG1P)
        else:
            raise ValueError(f"Unsupported field transform in checkpoint normalization: {name}={tr}")
    return modes


class PhysicalStateExportModule(nn.Module):
    """Export wrapper that consumes physical parameters and returns physical state."""

    def __init__(
        self,
        *,
        model: nn.Module,
        steps: int,
        normalization: Dict[str, Any],
        fields: list[str],
    ) -> None:
        super().__init__()
        self.model = model
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")

        self.register_buffer("param_mean", torch.tensor(normalization["param_mean"], dtype=torch.float32))
        self.register_buffer("param_std", torch.tensor(normalization["param_std"], dtype=torch.float32))
        self.register_buffer("state_mean", torch.tensor(normalization["state_mean"], dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(normalization["state_std"], dtype=torch.float32))

        mode_codes = _build_state_mode_codes(
            fields=fields,
            field_transforms=dict(normalization.get("field_transforms", {})),
        )
        if len(mode_codes) != int(self.state_mean.numel()):
            raise ValueError(
                "Checkpoint normalization/field mismatch: "
                f"{len(mode_codes)} field transforms vs {int(self.state_mean.numel())} state channels"
            )
        self.register_buffer("state_mode_codes", torch.tensor(mode_codes, dtype=torch.int64))

        self.param_zscore_eps = float(normalization["param_zscore_eps"])
        self.state_zscore_eps = float(normalization["state_zscore_eps"])
        self.signed_log1p_scale = float(normalization["signed_log1p_scale"])

    def _normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """Normalize physical parameter vectors using checkpoint statistics."""
        mean = self.param_mean.to(device=params.device, dtype=params.dtype)
        std = self.param_std.to(device=params.device, dtype=params.dtype)
        params_norm = (params - mean) / (std + self.param_zscore_eps)
        return torch.clamp(params_norm, -PARAM_NORM_CLIP_ABS, PARAM_NORM_CLIP_ABS)

    def _denormalize_state(self, state_norm: torch.Tensor) -> torch.Tensor:
        """Invert normalized model output back to physical state channels."""
        mean = self.state_mean.to(device=state_norm.device, dtype=state_norm.dtype).view(1, -1, 1, 1)
        std = self.state_std.to(device=state_norm.device, dtype=state_norm.dtype).view(1, -1, 1, 1)
        state = state_norm * (std + self.state_zscore_eps) + mean

        mode_codes = self.state_mode_codes.to(device=state.device).view(1, -1, 1, 1)
        none_mask = (mode_codes == FIELD_MODE_NONE).to(dtype=state.dtype)
        log10_mask = (mode_codes == FIELD_MODE_LOG10).to(dtype=state.dtype)
        signed_mask = (mode_codes == FIELD_MODE_SIGNED_LOG1P).to(dtype=state.dtype)

        state_log10 = torch.pow(
            10.0,
            torch.clamp(state, min=LOG10_INVERSE_CLIP_MIN, max=LOG10_INVERSE_CLIP_MAX),
        )
        state_signed = torch.sign(state) * torch.expm1(torch.abs(state)) * self.signed_log1p_scale
        return none_mask * state + log10_mask * state_log10 + signed_mask * state_signed

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Run normalized rollout and return denormalized physical state."""
        params_norm = self._normalize_params(params)
        state_norm = self.model(params_norm, steps=self.steps)
        return self._denormalize_state(state_norm)


def main() -> None:
    """Trace and save a TorchScript export with embedded normalization behavior."""
    _setup_warning_filters()

    model_dir = _resolve_model_dir(MODEL_DIR)
    ckpt_path = (model_dir / CHECKPOINT_NAME).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            f"Set MODEL_DIR to a valid run folder (for example under {(PROJECT_ROOT / 'models').resolve()})."
        )

    device = _resolve_device(DEVICE_MODE)
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=device)
    source_cfg_path = Path(str(ckpt["source_config_path"])).resolve()
    ensure_torch_harmonics_importable(source_cfg_path.parent)

    model_cfg = _dict_to_namespace(ckpt["model_config"])
    shape = dict(ckpt["shape"])
    state_chans = int(shape["C"])
    h = int(shape["H"])
    w = int(shape["W"])
    param_dim = int(len(ckpt["param_names"]))
    steps = int(ckpt["model_config"]["rollout_steps_at_default_time"])

    core_model = build_rollout_model(
        img_size=(h, w),
        state_chans=state_chans,
        param_dim=param_dim,
        cfg_model=model_cfg,
    )
    core_model.load_state_dict(ckpt["model_state"], strict=True)
    core_model.to(device=device)
    core_model.eval()

    export_model = PhysicalStateExportModule(
        model=core_model,
        steps=steps,
        normalization=dict(ckpt["normalization"]),
        fields=list(ckpt["fields"]),
    ).to(device=device).eval()

    with torch.inference_mode():
        param_mean = torch.tensor(ckpt["normalization"]["param_mean"], dtype=torch.float32, device=device)
        example_params = param_mean.unsqueeze(0).repeat(int(EXAMPLE_BATCH_SIZE), 1)
        traced = torch.jit.trace(
            export_model,
            example_params,
            strict=bool(STRICT_TRACE),
            check_trace=False,
        )
        traced = torch.jit.freeze(traced.eval())
        reference_out = export_model(example_params).detach().cpu()

    export_path = (model_dir / EXPORT_NAME).resolve()
    traced.save(str(export_path))

    # Verify on CPU to keep verification backend-agnostic.
    loaded = torch.jit.load(str(export_path), map_location=torch.device("cpu")).eval()
    with torch.inference_mode():
        loaded_out = loaded(example_params.detach().cpu()).detach().cpu()
    max_abs_diff = float((loaded_out - reference_out).abs().max().item())
    if max_abs_diff > 1.0e-4:
        raise RuntimeError(f"Export verification failed: max_abs_diff={max_abs_diff:.3e} exceeds tolerance")

    meta = {
        "export_format": "torchscript",
        "exported_from_checkpoint": str(ckpt_path),
        "exported_model_path": str(export_path),
        "device_used_for_export": str(device),
        "uses_physical_space_io": True,
        "fixed_rollout_steps": int(steps),
        "input": {
            "name": "params_physical",
            "shape": ["batch", param_dim],
            "dtype": "float32",
        },
        "output": {
            "name": "state_final_physical",
            "shape": ["batch", state_chans, h, w],
            "dtype": "float32",
            "fields": list(ckpt["fields"]),
        },
        "param_names": list(ckpt["param_names"]),
        "normalization_baked_in": dict(ckpt["normalization"]),
        "verification": {
            "max_abs_diff_vs_reference": max_abs_diff,
            "tolerance": 1.0e-4,
        },
    }

    meta_path = (model_dir / EXPORT_META_NAME).resolve()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved TorchScript export: {export_path}")
    print(f"Saved export metadata: {meta_path}")


if __name__ == "__main__":
    main()
