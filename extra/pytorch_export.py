"""Export a trained one-step GCMulator checkpoint as a physical-space TorchScript module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, Sequence
import warnings

import torch
import torch.nn as nn


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling import build_state_conditioned_transition_model, ensure_torch_harmonics_importable


EXPORT_NAME = "model_export.torchscript.pt"
EXPORT_META_NAME = "model_export.meta.json"
EXAMPLE_BATCH_SIZE = 1
STRICT_TRACE = True
SUPPRESS_KNOWN_EXPORT_WARNINGS = True

FIELD_MODE_NONE = 0
FIELD_MODE_LOG10 = 1
FIELD_MODE_SIGNED_LOG1P = 2


def _dict_to_namespace(obj: Any) -> Any:
    """Recursively convert nested dictionaries into namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(
            **{key: _dict_to_namespace(value) for key, value in obj.items()}
        )
    if isinstance(obj, list):
        return [_dict_to_namespace(value) for value in obj]
    return obj


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Export a trained GCMulator checkpoint to TorchScript"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Run directory containing best.pt")
    source.add_argument("--checkpoint", type=Path, help="Checkpoint path to export")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Export device",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit TorchScript output path",
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=None,
        help="Explicit metadata JSON output path",
    )
    return parser.parse_args()


def _resolve_checkpoint_path(*, run_dir: Path | None, checkpoint: Path | None) -> Path:
    """Resolve checkpoint path from CLI inputs."""
    if checkpoint is not None:
        resolved = checkpoint.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved
    if run_dir is None:
        raise ValueError("Either --run-dir or --checkpoint must be provided")
    resolved_run_dir = run_dir.resolve()
    ckpt_path = (resolved_run_dir / "best.pt").resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def _resolve_device(mode: str) -> torch.device:
    """Resolve export device."""
    normalized = str(mode).lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device=gpu but CUDA is unavailable")
        return torch.device("cuda")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device mode: {mode}")


def _setup_warning_filters() -> None:
    """Suppress known export-time warnings."""
    if not SUPPRESS_KNOWN_EXPORT_WARNINGS:
        return
    warnings.filterwarnings(
        "ignore",
        message=r"Casting complex values to real discards the imaginary part.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r".*Converting a tensor to a Python boolean might cause "
            r"the trace to be incorrect.*"
        ),
        category=torch.jit.TracerWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r".*Converting a tensor to a Python integer might cause "
            r"the trace to be incorrect.*"
        ),
        category=torch.jit.TracerWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"An output with one or more elements was resized since it had shape \[\],.*",
        category=UserWarning,
    )


def _build_state_mode_codes(
    *,
    fields: Sequence[str],
    field_transforms: Dict[str, Any],
) -> list[int]:
    """Encode string transform names as integer mode IDs."""
    mode_codes: list[int] = []
    for field_name in fields:
        transform = str(field_transforms.get(field_name, "none"))
        if transform == "none":
            mode_codes.append(FIELD_MODE_NONE)
        elif transform == "log10":
            mode_codes.append(FIELD_MODE_LOG10)
        elif transform == "signed_log1p":
            mode_codes.append(FIELD_MODE_SIGNED_LOG1P)
        else:
            raise ValueError(f"Unsupported field transform: {field_name}={transform}")
    return mode_codes


def _normalize_state_with_stats(
    state: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    mode_codes: torch.Tensor,
    zscore_eps: float,
    log10_eps: float,
    signed_log1p_scale: float,
) -> torch.Tensor:
    """Normalize a physical-space state tensor to the model space."""
    field_mode_none = 0
    field_mode_log10 = 1
    mean = mean.to(device=state.device, dtype=state.dtype)[None, :, None, None]
    std = std.to(device=state.device, dtype=state.dtype)[None, :, None, None]
    mode_codes = mode_codes.to(device=state.device)[None, :, None, None]

    state_none = state
    log10_input = torch.where(
        mode_codes == field_mode_log10,
        state,
        torch.ones_like(state),
    )
    torch._assert(
        torch.all(torch.isfinite(log10_input)),
        "log10 transform received non-finite values",
    )
    torch._assert(
        torch.all(log10_input > 0),
        "log10 transform requires strictly positive values",
    )
    state_log10 = torch.log10(torch.clamp(log10_input, min=float(log10_eps)))
    state_signed = torch.sign(state) * torch.log1p(
        torch.abs(state) / float(signed_log1p_scale)
    )
    transformed = torch.where(
        mode_codes == field_mode_none,
        state_none,
        torch.where(mode_codes == field_mode_log10, state_log10, state_signed),
    )
    return (transformed - mean) / (std + float(zscore_eps))


def _denormalize_state_with_stats(
    state_norm: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    mode_codes: torch.Tensor,
    zscore_eps: float,
    signed_log1p_scale: float,
) -> torch.Tensor:
    """Map normalized state output back to physical space."""
    field_mode_none = 0
    field_mode_log10 = 1
    mean = mean.to(device=state_norm.device, dtype=state_norm.dtype)[
        None, :, None, None
    ]
    std = std.to(device=state_norm.device, dtype=state_norm.dtype)[
        None, :, None, None
    ]
    transformed = state_norm * (std + float(zscore_eps)) + mean
    mode_codes = mode_codes.to(device=state_norm.device)[None, :, None, None]

    state_none = transformed
    state_log10 = torch.pow(
        torch.tensor(
            10.0,
            device=transformed.device,
            dtype=transformed.dtype,
        ),
        transformed,
    )
    state_signed = (
        torch.sign(transformed)
        * torch.expm1(torch.abs(transformed))
        * float(signed_log1p_scale)
    )
    return torch.where(
        mode_codes == field_mode_none,
        state_none,
        torch.where(mode_codes == field_mode_log10, state_log10, state_signed),
    )


class PhysicalStateExportModule(nn.Module):
    """TorchScript-exportable physical-I/O wrapper around the normalized core model."""

    def __init__(
        self,
        *,
        model: nn.Module,
        normalization: Dict[str, Any],
        input_fields: list[str],
        target_fields: list[str],
    ) -> None:
        """Store the normalized model and its physical-space normalization metadata."""
        super().__init__()
        self.model = model

        input_state = dict(normalization["input_state"])
        target_state = dict(normalization["target_state"])
        params = dict(normalization["params"])

        self.register_buffer(
            "param_mean",
            torch.tensor(params["mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "param_std",
            torch.tensor(params["std"], dtype=torch.float32),
        )
        self.register_buffer(
            "param_is_constant",
            torch.tensor(params["is_constant"], dtype=torch.bool),
        )

        self.register_buffer(
            "input_state_mean",
            torch.tensor(input_state["mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "input_state_std",
            torch.tensor(input_state["std"], dtype=torch.float32),
        )
        self.register_buffer(
            "input_state_mode_codes",
            torch.tensor(
                _build_state_mode_codes(
                    fields=input_fields,
                    field_transforms=dict(input_state.get("field_transforms", {})),
                ),
                dtype=torch.int64,
            ),
        )

        self.register_buffer(
            "target_state_mean",
            torch.tensor(target_state["mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "target_state_std",
            torch.tensor(target_state["std"], dtype=torch.float32),
        )
        self.register_buffer(
            "target_state_mode_codes",
            torch.tensor(
                _build_state_mode_codes(
                    fields=target_fields,
                    field_transforms=dict(target_state.get("field_transforms", {})),
                ),
                dtype=torch.int64,
            ),
        )

        self.input_state_zscore_eps = float(input_state["zscore_eps"])
        self.input_log10_eps = float(input_state["log10_eps"])
        self.input_signed_log1p_scale = float(input_state["signed_log1p_scale"])
        self.target_state_zscore_eps = float(target_state["zscore_eps"])
        self.target_signed_log1p_scale = float(target_state["signed_log1p_scale"])
        self.param_zscore_eps = float(params["zscore_eps"])
        self.base_param_dim = int(self.param_mean.numel())

    def _normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """Normalize physical conditioning parameters."""
        torch._assert(params.dim() == 2, "params must be rank-2")
        torch._assert(
            params.size(-1) == self.base_param_dim,
            "params width does not match checkpoint metadata",
        )
        mean = self.param_mean.to(device=params.device, dtype=params.dtype)
        std = self.param_std.to(device=params.device, dtype=params.dtype)
        out = (params - mean) / (std + self.param_zscore_eps)
        const_mask = self.param_is_constant.to(device=params.device)
        if torch.any(const_mask):
            out = out.masked_fill(const_mask.unsqueeze(0), 0.0)
        return out

    def _normalize_input_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize the full visible state to model space."""
        return _normalize_state_with_stats(
            state,
            mean=self.input_state_mean,
            std=self.input_state_std,
            mode_codes=self.input_state_mode_codes,
            zscore_eps=self.input_state_zscore_eps,
            log10_eps=self.input_log10_eps,
            signed_log1p_scale=self.input_signed_log1p_scale,
        )

    def _denormalize_target_state(self, state_norm: torch.Tensor) -> torch.Tensor:
        """Map normalized prognostic outputs back to physical space."""
        return _denormalize_state_with_stats(
            state_norm,
            mean=self.target_state_mean,
            std=self.target_state_std,
            mode_codes=self.target_state_mode_codes,
            zscore_eps=self.target_state_zscore_eps,
            signed_log1p_scale=self.target_signed_log1p_scale,
        )

    def example_state(
        self,
        *,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Construct a valid physical-space example tensor for tracing."""
        zero_norm = torch.zeros(
            (batch_size, int(self.input_state_mean.numel()), height, width),
            dtype=torch.float32,
            device=device,
        )
        return _denormalize_state_with_stats(
            zero_norm,
            mean=self.input_state_mean,
            std=self.input_state_std,
            mode_codes=self.input_state_mode_codes,
            zscore_eps=self.input_state_zscore_eps,
            signed_log1p_scale=self.input_signed_log1p_scale,
        )

    def forward(self, state0: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Run one physical-space transition step."""
        state0_norm = self._normalize_input_state(state0)
        params_norm = self._normalize_params(params)
        state1_norm = self.model(state0_norm, params_norm)
        return self._denormalize_target_state(state1_norm)


def main() -> None:
    """Export the selected checkpoint."""
    _setup_warning_filters()
    args = _parse_args()
    ckpt_path = _resolve_checkpoint_path(run_dir=args.run_dir, checkpoint=args.checkpoint)
    run_dir = ckpt_path.parent
    export_path = (
        args.output.resolve()
        if args.output is not None
        else (run_dir / EXPORT_NAME).resolve()
    )
    meta_path = (
        args.meta_output.resolve()
        if args.meta_output is not None
        else (run_dir / EXPORT_META_NAME).resolve()
    )

    device = _resolve_device(args.device)
    ensure_torch_harmonics_importable()
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=device)

    model_cfg = _dict_to_namespace(ckpt["model_config"])
    shape = dict(ckpt["shape"])
    input_fields = list(ckpt["input_fields"])
    target_fields = list(ckpt["target_fields"])
    residual_input_indices = [input_fields.index(field_name) for field_name in target_fields]

    core_model = build_state_conditioned_transition_model(
        img_size=(int(shape["H"]), int(shape["W"])),
        input_state_chans=int(shape["input_C"]),
        target_state_chans=int(shape["target_C"]),
        param_dim=int(len(ckpt["param_names"])),
        residual_input_indices=residual_input_indices,
        cfg_model=model_cfg,
        lat_order="north_to_south",
        lon_origin="0_to_2pi",
    )
    core_model.load_state_dict(ckpt["model_state"], strict=True)
    core_model.to(device=device).eval()

    with torch.inference_mode():
        bootstrap_model = PhysicalStateExportModule(
            model=core_model,
            normalization=dict(ckpt["normalization"]),
            input_fields=input_fields,
            target_fields=target_fields,
        ).to(device=device).eval()
        params_stats = dict(ckpt["normalization"]["params"])
        example_params = torch.tensor(
            params_stats["mean"],
            dtype=torch.float32,
            device=device,
        )[None, :]
        example_params = example_params.repeat(int(EXAMPLE_BATCH_SIZE), 1)
        example_state = bootstrap_model.example_state(
            batch_size=int(EXAMPLE_BATCH_SIZE),
            height=int(shape["H"]),
            width=int(shape["W"]),
            device=device,
        )
        traced_core = torch.jit.trace(
            core_model,
            (
                bootstrap_model._normalize_input_state(example_state),
                bootstrap_model._normalize_params(example_params),
            ),
            strict=bool(STRICT_TRACE),
            check_trace=False,
        )
        traced_core = torch.jit.freeze(traced_core.eval())
        export_model = PhysicalStateExportModule(
            model=traced_core,
            normalization=dict(ckpt["normalization"]),
            input_fields=input_fields,
            target_fields=target_fields,
        ).to(device=device).eval()
        exported = torch.jit.script(export_model)
        exported = torch.jit.freeze(exported.eval())
        reference_output = export_model(example_state, example_params).detach().cpu()

    export_path.parent.mkdir(parents=True, exist_ok=True)
    exported.save(str(export_path))

    loaded = torch.jit.load(str(export_path), map_location=torch.device("cpu")).eval()
    with torch.inference_mode():
        loaded_output = loaded(
            example_state.detach().cpu(),
            example_params.detach().cpu(),
        ).detach().cpu()
    max_abs_diff = float((loaded_output - reference_output).abs().max().item())
    if max_abs_diff > 1.0e-4:
        raise RuntimeError(f"Export verification failed: max_abs_diff={max_abs_diff:.3e}")

    meta = {
        "export_format": "torchscript",
        "checkpoint_path": str(ckpt_path),
        "export_path": str(export_path),
        "device": str(device),
        "input": {
            "state0": ["batch", int(shape["input_C"]), int(shape["H"]), int(shape["W"])],
            "params": ["batch", int(len(ckpt["param_names"]))],
            "fields": input_fields,
        },
        "output": {
            "state1": ["batch", int(shape["target_C"]), int(shape["H"]), int(shape["W"])],
            "fields": target_fields,
        },
        "param_names": list(ckpt["param_names"]),
        "normalization": dict(ckpt["normalization"]),
        "verification": {
            "max_abs_diff_vs_reference": max_abs_diff,
            "tolerance": 1.0e-4,
        },
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved TorchScript export: {export_path}")
    print(f"Saved export metadata: {meta_path}")


if __name__ == "__main__":
    main()
