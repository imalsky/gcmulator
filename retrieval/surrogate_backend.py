"""Contract inspection and batched Torch runtime helpers for surrogate retrieval.

This module does two separate jobs:

1. inspect a saved Torch export bundle and decide whether it satisfies the
   current direct-jump training contract
2. provide a CPU/GPU Torch inference runtime for valid direct-jump artifacts

The runtime intentionally accepts physical ``transition_days``. The export is
responsible for converting those durations into the normalized
``log10_transition_days`` feature used during training.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gcmulator.config import TRANSITION_TIME_NAME


FIELD_MODE_NONE = 0
FIELD_MODE_LOG10 = 1
FIELD_MODE_SIGNED_LOG1P = 2


def _load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed JSON if the file exists, else ``None``."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_required_export_meta(export_path: Path) -> Dict[str, Any]:
    """Load the metadata companion that makes the export bundle self-contained."""
    meta_path = export_path.with_name(
        export_path.name.replace(".torchscript.pt", ".meta.json")
    )
    meta = _load_optional_json(meta_path)
    if meta is None:
        raise FileNotFoundError(
            f"Export metadata not found: {meta_path}. "
            "The retrieval runtime expects a self-contained export bundle."
        )
    return meta


def _require_meta_mapping(meta: Dict[str, Any], *, key: str) -> Dict[str, Any]:
    """Require one dictionary-valued metadata block."""
    value = meta.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Export metadata is missing the `{key}` mapping")
    return dict(value)


def _require_meta_sequence(meta: Dict[str, Any], *, key: str) -> tuple[Any, ...]:
    """Require one list-like metadata block."""
    value = meta.get(key)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Export metadata is missing the `{key}` sequence")
    return tuple(value)


def _resolve_device(mode: str) -> torch.device:
    """Resolve the requested runtime device from explicit ``cpu`` / ``gpu`` modes."""
    normalized = str(mode).strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device_mode='gpu' but CUDA is unavailable")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device_mode: {mode!r}")


def _sync_device(device: torch.device) -> None:
    """Synchronize asynchronous accelerator work before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _build_state_mode_codes(
    *,
    fields: Sequence[str],
    field_transforms: Dict[str, Any],
) -> list[int]:
    """Encode per-field transform names into compact integer IDs."""
    mode_codes: list[int] = []
    for field_name in fields:
        transform = str(field_transforms.get(str(field_name), "none"))
        if transform == "none":
            mode_codes.append(FIELD_MODE_NONE)
        elif transform == "log10":
            mode_codes.append(FIELD_MODE_LOG10)
        elif transform == "signed_log1p":
            mode_codes.append(FIELD_MODE_SIGNED_LOG1P)
        else:
            raise ValueError(f"Unsupported field transform: {field_name}={transform}")
    return mode_codes


def _denormalize_state_with_stats(
    state_norm: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    mode_codes: torch.Tensor,
    zscore_eps: float,
    signed_log1p_scale: float,
) -> torch.Tensor:
    """Map normalized state tensors back into physical space."""
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
        torch.tensor(10.0, device=transformed.device, dtype=transformed.dtype),
        transformed,
    )
    state_signed = (
        torch.sign(transformed)
        * torch.expm1(torch.abs(transformed))
        * float(signed_log1p_scale)
    )
    return torch.where(
        mode_codes == FIELD_MODE_NONE,
        state_none,
        torch.where(mode_codes == FIELD_MODE_LOG10, state_log10, state_signed),
    )


@dataclass(frozen=True)
class SurrogateArtifactContract:
    """Summarize whether a saved surrogate can support the current retrieval contract."""

    export_path: Path
    checkpoint_path: Optional[Path]
    model_days: float
    export_format: str
    supported_devices: tuple[str, ...]
    forward_schema: str
    export_arg_count: int
    state_shape: tuple[int, int, int]
    param_names: tuple[str, ...]
    conditioning_names: tuple[str, ...]
    state_fields: tuple[str, ...]
    transition_days_conditioned: bool
    physical_io_forward: bool
    has_transition_time_stats: bool
    transition_time_name: Optional[str]
    sampling_saved_checkpoint_interval_days: Optional[float]
    sampling_live_transition_days_min: Optional[float]
    sampling_live_transition_days_max: Optional[float]
    sampling_live_transition_tolerance_fraction: Optional[float]
    direct_jump_torch_runtime_ready: bool
    torch_runtime_blockers: tuple[str, ...]
    blockers: tuple[str, ...]
    recommendations: tuple[str, ...]

    def to_json_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly payload."""
        return {
            "export_path": str(self.export_path),
            "checkpoint_path": (
                None if self.checkpoint_path is None else str(self.checkpoint_path)
            ),
            "model_days": float(self.model_days),
            "export_format": self.export_format,
            "supported_devices": list(self.supported_devices),
            "forward_schema": self.forward_schema,
            "export_arg_count": int(self.export_arg_count),
            "state_shape": list(self.state_shape),
            "param_names": list(self.param_names),
            "conditioning_names": list(self.conditioning_names),
            "state_fields": list(self.state_fields),
            "transition_days_conditioned": bool(self.transition_days_conditioned),
            "physical_io_forward": bool(self.physical_io_forward),
            "has_transition_time_stats": bool(self.has_transition_time_stats),
            "transition_time_name": self.transition_time_name,
            "sampling_saved_checkpoint_interval_days": self.sampling_saved_checkpoint_interval_days,
            "sampling_live_transition_days_min": self.sampling_live_transition_days_min,
            "sampling_live_transition_days_max": self.sampling_live_transition_days_max,
            "sampling_live_transition_tolerance_fraction": (
                self.sampling_live_transition_tolerance_fraction
            ),
            "direct_jump_torch_runtime_ready": bool(
                self.direct_jump_torch_runtime_ready
            ),
            "torch_runtime_blockers": list(self.torch_runtime_blockers),
            "blockers": list(self.blockers),
            "recommendations": list(self.recommendations),
        }


@dataclass(frozen=True)
class SurrogateRuntimeConfig:
    """User-tunable runtime settings for many-sample surrogate inference."""

    # Keep the runtime choice explicit for now: retrieval should be either an
    # intentional CPU run or an intentional GPU run.
    device_mode: str = "gpu"
    max_batch_size: int = 256
    pin_host_memory: bool = True
    prefer_channels_last: bool = True
    allow_tf32: bool = True


def inspect_surrogate_artifact(
    *,
    export_path: Path,
    checkpoint_path: Path | None = None,
    model_days: float = 100.0,
) -> SurrogateArtifactContract:
    """Inspect the saved artifact against the current direct-jump contract."""
    resolved_export = Path(export_path).resolve()
    if not resolved_export.is_file():
        raise FileNotFoundError(f"Export not found: {resolved_export}")
    if model_days <= 0.0:
        raise ValueError(f"model_days must be > 0, got {model_days}")

    export_model = torch.jit.load(str(resolved_export), map_location="cpu").eval()
    export_meta = _load_required_export_meta(resolved_export)

    resolved_checkpoint: Path | None = None
    checkpoint: Dict[str, Any] | None = None
    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path).resolve()
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")
        checkpoint = torch.load(resolved_checkpoint, map_location="cpu")

    solver = _require_meta_mapping(export_meta, key="solver")
    shape = _require_meta_mapping(export_meta, key="shape")
    sampling = _require_meta_mapping(export_meta, key="sampling")
    normalization = _require_meta_mapping(export_meta, key="normalization")
    input_meta = _require_meta_mapping(export_meta, key="input")
    output_meta = _require_meta_mapping(export_meta, key="output")
    runtime_hints = _require_meta_mapping(export_meta, key="runtime_hints")
    physical_io = _require_meta_mapping(export_meta, key="physical_io")
    param_names = tuple(
        str(value) for value in _require_meta_sequence(export_meta, key="param_names")
    )
    conditioning_names = tuple(
        str(value)
        for value in _require_meta_sequence(export_meta, key="conditioning_names")
    )
    state_fields = tuple(
        str(value) for value in _require_meta_sequence(input_meta, key="fields")
    )
    output_fields = tuple(
        str(value) for value in _require_meta_sequence(output_meta, key="fields")
    )
    if state_fields != output_fields:
        raise ValueError(
            "Export metadata input and output fields must be identical for the "
            f"autoregressive contract: input={list(state_fields)}, output={list(output_fields)}"
        )
    supported_devices = tuple(
        str(value).lower()
        for value in _require_meta_sequence(export_meta, key="supported_devices")
    )

    if checkpoint is not None:
        checkpoint_solver = dict(checkpoint["solver"])
        checkpoint_shape = dict(checkpoint["shape"])
        checkpoint_sampling = dict(checkpoint.get("sampling", {}))
        checkpoint_normalization = dict(checkpoint.get("normalization", {}))
        checkpoint_state_fields = tuple(str(value) for value in checkpoint["state_fields"])
        checkpoint_param_names = tuple(str(value) for value in checkpoint["param_names"])
        checkpoint_conditioning_names = tuple(
            str(value) for value in checkpoint.get("conditioning_names", ())
        )
        if solver != checkpoint_solver:
            raise ValueError("Export metadata solver block does not match the checkpoint")
        if shape != checkpoint_shape:
            raise ValueError("Export metadata shape block does not match the checkpoint")
        if sampling != checkpoint_sampling:
            raise ValueError("Export metadata sampling block does not match the checkpoint")
        if normalization != checkpoint_normalization:
            raise ValueError(
                "Export metadata normalization block does not match the checkpoint"
            )
        if state_fields != checkpoint_state_fields:
            raise ValueError("Export metadata state fields do not match the checkpoint")
        if param_names != checkpoint_param_names:
            raise ValueError("Export metadata param_names do not match the checkpoint")
        if conditioning_names != checkpoint_conditioning_names:
            raise ValueError(
                "Export metadata conditioning_names do not match the checkpoint"
            )

    has_transition_time_stats = "transition_time" in normalization
    transition_time_name: Optional[str] = None
    if has_transition_time_stats:
        transition_time = dict(normalization["transition_time"])
        names = tuple(str(value) for value in transition_time.get("param_names", ()))
        transition_time_name = names[0] if names else None

    sampling_saved_checkpoint_interval_days = sampling.get("saved_checkpoint_interval_days")
    sampling_live_transition_days_min = sampling.get("live_transition_days_min")
    sampling_live_transition_days_max = sampling.get("live_transition_days_max")
    sampling_live_transition_tolerance_fraction = sampling.get(
        "live_transition_tolerance_fraction"
    )

    export_arg_count = len(export_model.forward.schema.arguments) - 1
    transition_days_conditioned = export_arg_count == 3

    expected_conditioning_names = param_names + (TRANSITION_TIME_NAME,)
    torch_runtime_blockers: List[str] = []
    recommendations: List[str] = []
    artifact_kind = str(export_meta.get("artifact_kind", ""))
    export_format = str(export_meta.get("export_format", ""))
    input_state_shape = tuple(input_meta.get("state0", ()))
    output_state_shape = tuple(output_meta.get("state1", ()))
    params_shape = tuple(input_meta.get("params", ()))
    transition_days_shape = tuple(input_meta.get("transition_days", ()))
    normalization_state_fields = tuple(
        str(value)
        for value in dict(normalization.get("state", {})).get("field_names", ())
    )
    physical_io_forward = all(
        bool(physical_io.get(name, False))
        for name in ("state0", "params", "transition_days", "state1")
    )

    if artifact_kind != "direct_jump_physical_state_transition":
        torch_runtime_blockers.append(
            "The export metadata `artifact_kind` is not the expected direct-jump "
            "physical-state transition bundle."
        )
        recommendations.append(
            "Re-export the surrogate with the current physical-space Torch export script."
        )

    if export_format != "torchscript":
        torch_runtime_blockers.append(
            f"The export metadata declares format `{export_format}`, but retrieval "
            "currently expects a TorchScript bundle."
        )
        recommendations.append(
            "Re-export the surrogate with the TorchScript export entry point."
        )

    if supported_devices != ("cpu", "gpu"):
        torch_runtime_blockers.append(
            "The export metadata `supported_devices` must be exactly `['cpu', 'gpu']` "
            "for the current runtime surface."
        )
        recommendations.append(
            "Regenerate the export metadata so the supported runtime devices stay explicit."
        )

    if str(runtime_hints.get("transition_time_feature")) != TRANSITION_TIME_NAME:
        torch_runtime_blockers.append(
            "The export metadata does not declare the current "
            f"`{TRANSITION_TIME_NAME}` transition-time feature."
        )
        recommendations.append(
            "Regenerate the export metadata from a checkpoint trained with the "
            "current direct-jump conditioning contract."
        )

    if not physical_io_forward:
        torch_runtime_blockers.append(
            "The export metadata does not mark all inputs/outputs as physical-space "
            "tensors (`state0`, `params`, `transition_days`, `state1`)."
        )
        recommendations.append(
            "Re-export the surrogate with the physical-state wrapper enabled."
        )
    if input_state_shape != (
        "batch",
        int(shape["C"]),
        int(shape["H"]),
        int(shape["W"]),
    ):
        torch_runtime_blockers.append(
            "The export metadata `input.state0` shape does not match the saved "
            "shape contract."
        )
        recommendations.append(
            "Regenerate the export metadata from the same TorchScript artifact."
        )
    if params_shape != ("batch", int(len(param_names))):
        torch_runtime_blockers.append(
            "The export metadata `input.params` shape does not match `param_names`."
        )
        recommendations.append(
            "Regenerate the export metadata so parameter dimensions stay explicit."
        )
    if transition_days_shape != ("batch",):
        torch_runtime_blockers.append(
            "The export metadata `input.transition_days` shape must be `['batch']`."
        )
        recommendations.append(
            "Regenerate the export metadata so the time-jump input contract stays explicit."
        )
    if output_state_shape != (
        "batch",
        int(shape["C"]),
        int(shape["H"]),
        int(shape["W"]),
    ):
        torch_runtime_blockers.append(
            "The export metadata `output.state1` shape does not match the saved "
            "shape contract."
        )
        recommendations.append(
            "Regenerate the export metadata from the same TorchScript artifact."
        )
    if normalization_state_fields and state_fields != normalization_state_fields:
        torch_runtime_blockers.append(
            "The export metadata state field list does not match "
            "`normalization.state.field_names`."
        )
        recommendations.append(
            "Regenerate the export so the physical field ordering is self-consistent."
        )

    if not has_transition_time_stats:
        torch_runtime_blockers.append(
            "The export metadata normalization does not include `transition_time`, so the export "
            "is missing the current variable-jump training contract."
        )
        recommendations.append(
            "Re-export from a checkpoint that stores `normalization.transition_time`."
        )

    if transition_time_name is not None and transition_time_name != TRANSITION_TIME_NAME:
        torch_runtime_blockers.append(
            f"The export transition-time feature is named `{transition_time_name}`, but the "
            f"current training contract expects `{TRANSITION_TIME_NAME}`."
        )
        recommendations.append(
            "Re-export the surrogate from a checkpoint produced by the current training code."
        )

    if conditioning_names != expected_conditioning_names:
        torch_runtime_blockers.append(
            "The export conditioning vector does not match the direct-jump contract: "
            f"expected {list(expected_conditioning_names)!r}, got {list(conditioning_names)!r}."
        )
        recommendations.append(
            "Export a checkpoint whose conditioning vector appends "
            f"`{TRANSITION_TIME_NAME}` after the physical parameters."
        )

    if not transition_days_conditioned:
        torch_runtime_blockers.append(
            "The export forward signature does not accept `transition_days`, "
            "so it does not satisfy the direct-jump runtime contract."
        )
        recommendations.append(
            "Train/export a direct-jump model with explicit `transition_days` conditioning."
        )

    if sampling_saved_checkpoint_interval_days is None:
        torch_runtime_blockers.append(
            "The export sampling metadata does not record `saved_checkpoint_interval_days`, "
            "so the checkpoint-sequence training cadence is unknown."
        )
        recommendations.append(
            "Re-export the surrogate from a checkpoint produced by the sequence-based live-sampling "
            "training pipeline."
        )

    if sampling_live_transition_days_min is None or sampling_live_transition_days_max is None:
        torch_runtime_blockers.append(
            "The export sampling metadata does not record the live `transition_days` range, "
            "so there is no evidence this artifact was trained for flexible direct-jump horizons."
        )
        recommendations.append(
            "Train with live direct-jump sampling and store the live day range in the export metadata."
        )

    if transition_days_conditioned and "transition_days" not in input_meta:
        torch_runtime_blockers.append(
            "The saved export metadata does not advertise `transition_days` even though "
            "the runtime signature suggests time conditioning."
        )
        recommendations.append(
            "Regenerate the export metadata from the same artifact so the contract is unambiguous."
        )

    direct_jump_torch_ready = not torch_runtime_blockers

    return SurrogateArtifactContract(
        export_path=resolved_export,
        checkpoint_path=resolved_checkpoint,
        model_days=float(model_days),
        export_format=export_format,
        supported_devices=supported_devices,
        forward_schema=str(export_model.forward.schema),
        export_arg_count=int(export_arg_count),
        state_shape=(int(shape["C"]), int(shape["H"]), int(shape["W"])),
        param_names=param_names,
        conditioning_names=conditioning_names,
        state_fields=state_fields,
        transition_days_conditioned=bool(transition_days_conditioned),
        physical_io_forward=bool(physical_io_forward),
        has_transition_time_stats=bool(has_transition_time_stats),
        transition_time_name=transition_time_name,
        sampling_saved_checkpoint_interval_days=(
            None
            if sampling_saved_checkpoint_interval_days is None
            else float(sampling_saved_checkpoint_interval_days)
        ),
        sampling_live_transition_days_min=(
            None
            if sampling_live_transition_days_min is None
            else float(sampling_live_transition_days_min)
        ),
        sampling_live_transition_days_max=(
            None
            if sampling_live_transition_days_max is None
            else float(sampling_live_transition_days_max)
        ),
        sampling_live_transition_tolerance_fraction=(
            None
            if sampling_live_transition_tolerance_fraction is None
            else float(sampling_live_transition_tolerance_fraction)
        ),
        direct_jump_torch_runtime_ready=bool(direct_jump_torch_ready),
        torch_runtime_blockers=tuple(torch_runtime_blockers),
        blockers=tuple(torch_runtime_blockers),
        recommendations=tuple(recommendations),
    )


def assert_direct_jump_torch_runtime_ready(contract: SurrogateArtifactContract) -> None:
    """Raise a readable error when the artifact cannot run in the Torch backend."""
    if contract.direct_jump_torch_runtime_ready:
        return
    joined = "\n".join(f"- {item}" for item in contract.torch_runtime_blockers)
    raise RuntimeError(
        "The surrogate artifact is not runnable in the direct-jump Torch retrieval backend.\n"
        f"{joined}"
    )


class TorchSurrogateRuntime:
    """Batched Torch inference wrapper for many-sample direct-jump retrieval calls.

    The runtime performs inference through the exported TorchScript bundle. The
    training checkpoint is optional and only used when you want strict
    export-versus-checkpoint validation.
    """

    def __init__(
        self,
        *,
        export_path: Path,
        checkpoint_path: Path | None = None,
        runtime_config: SurrogateRuntimeConfig | None = None,
        model_days: float = 100.0,
    ) -> None:
        """Load and validate one export/checkpoint pair."""
        self.config = runtime_config or SurrogateRuntimeConfig()
        self.contract = inspect_surrogate_artifact(
            export_path=export_path,
            checkpoint_path=checkpoint_path,
            model_days=model_days,
        )
        assert_direct_jump_torch_runtime_ready(self.contract)

        self.device = _resolve_device(self.config.device_mode)
        self.export_meta = _load_required_export_meta(self.contract.export_path)
        loaded = torch.jit.load(
            str(self.contract.export_path),
            map_location="cpu",
        ).eval()
        try:
            loaded = torch.jit.optimize_for_inference(loaded)
        except RuntimeError:
            # Some TorchScript graphs cannot be further optimized; the frozen
            # export still remains valid for retrieval.
            pass
        self.model = loaded.to(device=self.device).eval()

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(self.config.allow_tf32)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.allow_tf32 = bool(self.config.allow_tf32)
                torch.backends.cudnn.benchmark = True

        shape = dict(self.export_meta["shape"])
        self.input_shape = (int(shape["C"]), int(shape["H"]), int(shape["W"]))
        self.param_dim = int(len(self.export_meta["param_names"]))

        # Example tensors come from the export metadata rather than hard-coded
        # constants so smoke checks stay valid even when field transforms change.
        self._example_state_cpu = self._build_example_state()
        self._example_params_cpu = torch.tensor(
            self.export_meta["normalization"]["params"]["mean"],
            dtype=torch.float32,
        ).reshape(1, -1)
        transition_mean_log10 = float(
            self.export_meta["normalization"]["transition_time"]["mean"][0]
        )
        self._example_transition_days_cpu = torch.tensor(
            [10.0 ** transition_mean_log10],
            dtype=torch.float32,
        )

    def _build_example_state(self) -> torch.Tensor:
        """Construct one valid physical-space state from normalization statistics."""
        normalization = dict(self.export_meta["normalization"])
        state = dict(normalization["state"])
        fields = list(self.export_meta["input"]["fields"])
        zero_norm = torch.zeros((1, *self.input_shape), dtype=torch.float32)
        return _denormalize_state_with_stats(
            zero_norm,
            mean=torch.tensor(state["mean"], dtype=torch.float32),
            std=torch.tensor(state["std"], dtype=torch.float32),
            mode_codes=torch.tensor(
                _build_state_mode_codes(
                    fields=fields,
                    field_transforms=dict(state.get("field_transforms", {})),
                ),
                dtype=torch.int64,
            ),
            zscore_eps=float(state["zscore_eps"]),
            signed_log1p_scale=float(state["signed_log1p_scale"]),
        )

    def example_inputs(
        self,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return valid example inputs for smoke tests and throughput checks."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return (
            self._example_state_cpu.repeat(int(batch_size), 1, 1, 1),
            self._example_params_cpu.repeat(int(batch_size), 1),
            self._example_transition_days_cpu.repeat(int(batch_size)),
        )

    def _broadcast_rows(self, tensor: torch.Tensor, *, batch_size: int, name: str) -> torch.Tensor:
        """Broadcast a single row to ``batch_size`` or validate an existing batch."""
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.repeat(int(batch_size), *([1] * (tensor.ndim - 1)))
        raise ValueError(
            f"{name} batch dimension must be 1 or {batch_size}, got {tuple(tensor.shape)}"
        )

    def _prepare_inputs(
        self,
        *,
        state0: np.ndarray | torch.Tensor,
        params: np.ndarray | torch.Tensor,
        transition_days: np.ndarray | torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert retrieval inputs to validated CPU tensors."""
        state0_tensor = torch.as_tensor(state0, dtype=torch.float32)
        if state0_tensor.ndim == 3:
            state0_tensor = state0_tensor.unsqueeze(0)
        if state0_tensor.ndim != 4:
            raise ValueError(
                "state0 must be [C,H,W] or [N,C,H,W], "
                f"got {tuple(state0_tensor.shape)}"
            )
        if tuple(state0_tensor.shape[1:]) != self.input_shape:
            raise ValueError(
                f"state0 shape must end with {self.input_shape}, got {tuple(state0_tensor.shape)}"
            )

        params_tensor = torch.as_tensor(params, dtype=torch.float32)
        if params_tensor.ndim == 1:
            params_tensor = params_tensor.unsqueeze(0)
        if params_tensor.ndim != 2 or int(params_tensor.shape[1]) != self.param_dim:
            raise ValueError(
                f"params must be [P] or [N,P] with P={self.param_dim}, "
                f"got {tuple(params_tensor.shape)}"
            )

        transition_days_tensor = torch.as_tensor(transition_days, dtype=torch.float32)
        if transition_days_tensor.ndim == 0:
            transition_days_tensor = transition_days_tensor.reshape(1)
        if transition_days_tensor.ndim != 1:
            raise ValueError(
                "transition_days must be scalar or [N], "
                f"got {tuple(transition_days_tensor.shape)}"
            )
        if not torch.isfinite(transition_days_tensor).all():
            raise ValueError("transition_days contains non-finite values")
        if torch.any(transition_days_tensor <= 0):
            raise ValueError("transition_days must be strictly positive")

        batch_size = int(state0_tensor.shape[0])
        params_tensor = self._broadcast_rows(
            params_tensor,
            batch_size=batch_size,
            name="params",
        )
        transition_days_tensor = self._broadcast_rows(
            transition_days_tensor.unsqueeze(1),
            batch_size=batch_size,
            name="transition_days",
        ).reshape(-1)

        return (
            state0_tensor.contiguous(),
            params_tensor.contiguous(),
            transition_days_tensor.contiguous(),
        )

    def _move_batch(self, tensor: torch.Tensor, *, is_state: bool) -> torch.Tensor:
        """Move one batch to the runtime device with overlap-friendly settings."""
        batch = tensor
        if self.device.type == "cuda" and batch.device.type == "cpu" and self.config.pin_host_memory:
            batch = batch.pin_memory()
        if is_state and self.config.prefer_channels_last and batch.ndim == 4:
            batch = batch.contiguous(memory_format=torch.channels_last)
        return batch.to(
            device=self.device,
            non_blocking=(self.device.type == "cuda" and batch.device.type == "cpu"),
        )

    def predict(
        self,
        *,
        state0: np.ndarray | torch.Tensor,
        params: np.ndarray | torch.Tensor,
        transition_days: np.ndarray | torch.Tensor | float,
        batch_size: int | None = None,
        return_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Run one batched direct-jump prediction in physical space."""
        state0_cpu, params_cpu, transition_days_cpu = self._prepare_inputs(
            state0=state0,
            params=params,
            transition_days=transition_days,
        )
        effective_batch_size = (
            int(self.config.max_batch_size)
            if batch_size is None
            else int(batch_size)
        )
        if effective_batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        outputs: List[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, int(state0_cpu.shape[0]), effective_batch_size):
                stop = min(int(state0_cpu.shape[0]), start + effective_batch_size)
                pred_batch = self.model(
                    self._move_batch(state0_cpu[start:stop], is_state=True),
                    self._move_batch(params_cpu[start:stop], is_state=False),
                    self._move_batch(transition_days_cpu[start:stop], is_state=False),
                )
                if return_numpy:
                    outputs.append(pred_batch.detach().cpu())
                else:
                    outputs.append(pred_batch.detach())

        combined = torch.cat(outputs, dim=0)
        if return_numpy:
            return combined.numpy()
        return combined

    def benchmark(
        self,
        *,
        batch_size: int,
        repeats: int = 10,
        warmup: int = 2,
    ) -> Dict[str, float]:
        """Benchmark one repeated forward pass using valid example inputs."""
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        state0, params, transition_days = self.example_inputs(batch_size=batch_size)
        with torch.inference_mode():
            for _ in range(int(warmup)):
                _ = self.predict(
                    state0=state0,
                    params=params,
                    transition_days=transition_days,
                    batch_size=batch_size,
                    return_numpy=False,
                )
            _sync_device(self.device)
            times: List[float] = []
            for _ in range(int(repeats)):
                _sync_device(self.device)
                start = time.perf_counter()
                _ = self.predict(
                    state0=state0,
                    params=params,
                    transition_days=transition_days,
                    batch_size=batch_size,
                    return_numpy=False,
                )
                _sync_device(self.device)
                times.append(time.perf_counter() - start)

        mean_seconds = float(np.mean(times))
        return {
            "device": str(self.device),
            "batch_size": int(batch_size),
            "repeats": int(repeats),
            "mean_seconds": mean_seconds,
            "samples_per_second": float(batch_size) / mean_seconds,
        }
