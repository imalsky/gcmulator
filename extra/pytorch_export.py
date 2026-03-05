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

from src.modeling import build_state_conditioned_rollout_model, ensure_torch_harmonics_importable

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
    """Export wrapper for state-conditioned rollout in physical space."""

    def __init__(
        self,
        *,
        model: nn.Module,
        steps: int,
        normalization: Dict[str, Any],
        fields: list[str],
        conditioning_names: list[str],
        transition_days_norm: Dict[str, Any],
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
        self.conditioning_names = [str(name) for name in conditioning_names]
        self.base_param_dim = int(self.param_mean.numel())
        self.expected_conditioning_dim = int(len(self.conditioning_names))
        self.use_transition_days_conditioning = (
            self.expected_conditioning_dim == self.base_param_dim + 1
            and bool(self.conditioning_names)
            and self.conditioning_names[-1] == "transition_days"
        )
        if self.expected_conditioning_dim not in {self.base_param_dim, self.base_param_dim + 1}:
            raise ValueError(
                "Unsupported conditioning schema in export module: "
                f"base_param_dim={self.base_param_dim}, conditioning_names={self.conditioning_names}"
            )
        if self.expected_conditioning_dim == self.base_param_dim + 1 and not self.use_transition_days_conditioning:
            raise ValueError(
                "Only trailing transition_days augmentation is supported for conditioning_names, "
                f"got {self.conditioning_names}."
            )
        self.transition_days_mean = float(transition_days_norm.get("mean", 0.0))
        self.transition_days_std = float(transition_days_norm.get("std", 1.0))
        self.transition_days_zscore_eps = float(transition_days_norm.get("zscore_eps", self.param_zscore_eps))
        self.transition_days_is_constant = bool(transition_days_norm.get("is_constant", True))

    def _normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """Normalize physical parameter vectors using checkpoint statistics."""
        if params.ndim != 2 or int(params.shape[-1]) != self.base_param_dim:
            raise ValueError(
                "params must be [B,P] with P equal to checkpoint param_mean length. "
                f"Got {tuple(params.shape)} expected P={self.base_param_dim}."
            )
        mean = self.param_mean.to(device=params.device, dtype=params.dtype)
        std = self.param_std.to(device=params.device, dtype=params.dtype)
        params_norm = (params - mean) / (std + self.param_zscore_eps)
        return torch.clamp(params_norm, -PARAM_NORM_CLIP_ABS, PARAM_NORM_CLIP_ABS)

    def _normalize_transition_days(self, transition_days: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        """Normalize physical transition durations to the model conditioning scale."""
        if transition_days.ndim != 1:
            raise ValueError(f"transition_days must be [B], got {tuple(transition_days.shape)}")
        td = transition_days.to(dtype=dtype)
        if self.transition_days_is_constant:
            td_norm = torch.zeros_like(td)
        else:
            td_norm = (td - self.transition_days_mean) / (self.transition_days_std + self.transition_days_zscore_eps)
        return torch.clamp(td_norm, -PARAM_NORM_CLIP_ABS, PARAM_NORM_CLIP_ABS)

    def _build_conditioning(self, *, params: torch.Tensor, transition_days: torch.Tensor) -> torch.Tensor:
        """Assemble normalized conditioning vector expected by the core model."""
        params_norm = self._normalize_params(params)
        if not self.use_transition_days_conditioning:
            return params_norm
        td_norm = self._normalize_transition_days(transition_days, dtype=params_norm.dtype)
        return torch.cat([params_norm, td_norm.view(-1, 1)], dim=1)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize physical state channels to the model space."""
        mean = self.state_mean.to(device=state.device, dtype=state.dtype).view(1, -1, 1, 1)
        std = self.state_std.to(device=state.device, dtype=state.dtype).view(1, -1, 1, 1)

        mode_codes = self.state_mode_codes.to(device=state.device).view(1, -1, 1, 1)
        none_mask = (mode_codes == FIELD_MODE_NONE).to(dtype=state.dtype)
        log10_mask = (mode_codes == FIELD_MODE_LOG10).to(dtype=state.dtype)
        signed_mask = (mode_codes == FIELD_MODE_SIGNED_LOG1P).to(dtype=state.dtype)

        # Forward transforms before z-score in training; invert ordering here by
        # applying transform in physical space then z-scoring.
        s = state
        s_log10 = torch.log10(torch.clamp(s, min=1.0e-30))
        s_signed = torch.sign(s) * torch.log1p(torch.abs(s) / self.signed_log1p_scale)
        transformed = none_mask * s + log10_mask * s_log10 + signed_mask * s_signed
        return (transformed - mean) / (std + self.state_zscore_eps)

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

    def forward(self, state0: torch.Tensor, params: torch.Tensor, transition_days: torch.Tensor) -> torch.Tensor:
        """Run normalized rollout from physical state0 and return physical state."""
        state0_norm = self._normalize_state(state0)
        conditioning_norm = self._build_conditioning(params=params, transition_days=transition_days)
        state_norm = self.model(state0_norm, conditioning_norm, steps=self.steps)
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
    ckpt_geometry = dict(ckpt.get("geometry", {}))
    lat_order = str(ckpt_geometry.get("lat_order", "north_to_south"))
    lon_origin = str(ckpt_geometry.get("lon_origin", "0_to_2pi"))
    shape = dict(ckpt["shape"])
    state_chans = int(shape["C"])
    h = int(shape["H"])
    w = int(shape["W"])
    param_dim = int(len(ckpt["param_names"]))
    conditioning_names = list(ckpt.get("conditioning_names", ckpt["param_names"]))
    conditioning_dim = int(len(conditioning_names))
    steps = 1
    transition_jump_steps = int(ckpt["model_config"].get("transition_jump_steps", 1))

    core_model = build_state_conditioned_rollout_model(
        img_size=(h, w),
        state_chans=state_chans,
        param_dim=conditioning_dim,
        cfg_model=model_cfg,
        lat_order=lat_order,
        lon_origin=lon_origin,
    )
    core_model.load_state_dict(ckpt["model_state"], strict=True)
    core_model.to(device=device)
    core_model.eval()

    export_model = PhysicalStateExportModule(
        model=core_model,
        steps=steps,
        normalization=dict(ckpt["normalization"]),
        fields=list(ckpt["fields"]),
        conditioning_names=conditioning_names,
        transition_days_norm=dict(ckpt.get("transition_days_norm", {})),
    ).to(device=device).eval()

    with torch.inference_mode():
        param_mean = torch.tensor(ckpt["normalization"]["param_mean"], dtype=torch.float32, device=device)
        state_mean = torch.tensor(ckpt["normalization"]["state_mean"], dtype=torch.float32, device=device)
        example_params = param_mean.unsqueeze(0).repeat(int(EXAMPLE_BATCH_SIZE), 1)
        if "transition_days_norm" in ckpt and "mean" in dict(ckpt.get("transition_days_norm", {})):
            transition_days_default = float(dict(ckpt["transition_days_norm"])["mean"])
        else:
            dt_seconds = float(dict(ckpt.get("solver", {})).get("dt_seconds", 240.0))
            transition_days_default = float(transition_jump_steps) * dt_seconds / 86400.0
        example_transition_days = torch.full(
            (int(EXAMPLE_BATCH_SIZE),),
            fill_value=float(transition_days_default),
            dtype=torch.float32,
            device=device,
        )
        example_state = state_mean.view(1, state_chans, 1, 1).repeat(int(EXAMPLE_BATCH_SIZE), 1, h, w)
        traced = torch.jit.trace(
            export_model,
            (example_state, example_params, example_transition_days),
            strict=bool(STRICT_TRACE),
            check_trace=False,
        )
        traced = torch.jit.freeze(traced.eval())
        reference_out = export_model(example_state, example_params, example_transition_days).detach().cpu()

    export_path = (model_dir / EXPORT_NAME).resolve()
    traced.save(str(export_path))

    # Verify on CPU to keep verification backend-agnostic.
    loaded = torch.jit.load(str(export_path), map_location=torch.device("cpu")).eval()
    with torch.inference_mode():
        loaded_out = loaded(
            example_state.detach().cpu(),
            example_params.detach().cpu(),
            example_transition_days.detach().cpu(),
        ).detach().cpu()
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
        "transition_jump_steps": int(transition_jump_steps),
        "input": {
            "name": "state0_params_transition_days",
            "shape": {
                "state0": ["batch", state_chans, h, w],
                "params": ["batch", param_dim],
                "transition_days": ["batch"],
            },
            "dtype": {"state0": "float32", "params": "float32", "transition_days": "float32"},
        },
        "output": {
            "name": "state_rollout_physical",
            "shape": ["batch", state_chans, h, w],
            "dtype": "float32",
            "fields": list(ckpt["fields"]),
        },
        "param_names": list(ckpt["param_names"]),
        "conditioning_names": conditioning_names,
        "conditioning_dim": int(conditioning_dim),
        "transition_days_norm": dict(ckpt.get("transition_days_norm", {})),
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
