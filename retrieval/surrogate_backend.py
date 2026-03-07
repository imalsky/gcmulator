"""Contract inspection and batched Torch runtime helpers for surrogate retrieval.

This module does two separate jobs:

1. inspect a saved export/checkpoint pair and decide whether it satisfies the
   current direct-jump training contract
2. provide a GPU-aware Torch inference runtime for valid direct-jump artifacts

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

from config import TRANSITION_TIME_NAME


FIELD_MODE_NONE = 0
FIELD_MODE_LOG10 = 1
FIELD_MODE_SIGNED_LOG1P = 2


def _load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed JSON if the file exists, else ``None``."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_device(mode: str) -> torch.device:
    """Resolve the requested runtime device."""
    normalized = str(mode).strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device_mode='gpu' but CUDA is unavailable")
        return torch.device("cuda")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
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
    checkpoint_path: Path
    model_days: float
    export_format: str
    forward_schema: str
    export_arg_count: int
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    param_names: tuple[str, ...]
    conditioning_names: tuple[str, ...]
    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    transition_days_conditioned: bool
    has_transition_time_stats: bool
    transition_time_name: Optional[str]
    sampling_transition_days_min: Optional[float]
    sampling_transition_days_max: Optional[float]
    fixed_step_days: float
    rollout_steps_for_model_days: int
    direct_jump_torch_runtime_ready: bool
    gpu_native_jax_retrieval_ready: bool
    torch_runtime_blockers: tuple[str, ...]
    jax_runtime_blockers: tuple[str, ...]
    blockers: tuple[str, ...]
    recommendations: tuple[str, ...]

    def to_json_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly payload."""
        return {
            "export_path": str(self.export_path),
            "checkpoint_path": str(self.checkpoint_path),
            "model_days": float(self.model_days),
            "export_format": self.export_format,
            "forward_schema": self.forward_schema,
            "export_arg_count": int(self.export_arg_count),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "param_names": list(self.param_names),
            "conditioning_names": list(self.conditioning_names),
            "input_fields": list(self.input_fields),
            "output_fields": list(self.output_fields),
            "transition_days_conditioned": bool(self.transition_days_conditioned),
            "has_transition_time_stats": bool(self.has_transition_time_stats),
            "transition_time_name": self.transition_time_name,
            "sampling_transition_days_min": self.sampling_transition_days_min,
            "sampling_transition_days_max": self.sampling_transition_days_max,
            "fixed_step_days": float(self.fixed_step_days),
            "rollout_steps_for_model_days": int(self.rollout_steps_for_model_days),
            "direct_jump_torch_runtime_ready": bool(
                self.direct_jump_torch_runtime_ready
            ),
            "gpu_native_jax_retrieval_ready": bool(
                self.gpu_native_jax_retrieval_ready
            ),
            "torch_runtime_blockers": list(self.torch_runtime_blockers),
            "jax_runtime_blockers": list(self.jax_runtime_blockers),
            "blockers": list(self.blockers),
            "recommendations": list(self.recommendations),
        }


@dataclass(frozen=True)
class SurrogateRuntimeConfig:
    """User-tunable runtime settings for many-sample surrogate inference."""

    device_mode: str = "auto"
    max_batch_size: int = 256
    pin_host_memory: bool = True
    prefer_channels_last: bool = True
    allow_tf32: bool = True


def inspect_surrogate_artifact(
    *,
    export_path: Path,
    checkpoint_path: Path,
    model_days: float = 100.0,
) -> SurrogateArtifactContract:
    """Inspect the saved artifact against the current direct-jump contract."""
    resolved_export = Path(export_path).resolve()
    resolved_checkpoint = Path(checkpoint_path).resolve()
    if not resolved_export.is_file():
        raise FileNotFoundError(f"Export not found: {resolved_export}")
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")
    if model_days <= 0.0:
        raise ValueError(f"model_days must be > 0, got {model_days}")

    export_model = torch.jit.load(str(resolved_export), map_location="cpu").eval()
    checkpoint: Dict[str, Any] = torch.load(resolved_checkpoint, map_location="cpu")
    export_meta = _load_optional_json(
        resolved_export.with_name(
            resolved_export.name.replace(".torchscript.pt", ".meta.json")
        )
    )

    solver = dict(checkpoint["solver"])
    shape = dict(checkpoint["shape"])
    sampling = dict(checkpoint.get("sampling", {}))
    normalization = dict(checkpoint.get("normalization", {}))
    input_fields = tuple(str(value) for value in checkpoint["input_fields"])
    output_fields = tuple(str(value) for value in checkpoint["target_fields"])
    param_names = tuple(str(value) for value in checkpoint["param_names"])
    conditioning_names = tuple(
        str(value) for value in checkpoint.get("conditioning_names", ())
    )

    has_transition_time_stats = "transition_time" in normalization
    transition_time_name: Optional[str] = None
    if has_transition_time_stats:
        transition_time = dict(normalization["transition_time"])
        names = tuple(str(value) for value in transition_time.get("param_names", ()))
        transition_time_name = names[0] if names else None

    sampling_transition_days_min = sampling.get("transition_jump_days_min")
    sampling_transition_days_max = sampling.get("transition_jump_days_max")

    export_arg_count = len(export_model.forward.schema.arguments) - 1
    transition_days_conditioned = export_arg_count == 3
    fixed_step_days = float(solver["dt_seconds"]) / 86400.0
    rollout_steps = int(round(float(model_days) / fixed_step_days))

    expected_conditioning_names = param_names + (TRANSITION_TIME_NAME,)
    torch_runtime_blockers: List[str] = []
    jax_runtime_blockers: List[str] = []
    recommendations: List[str] = []

    if not has_transition_time_stats:
        torch_runtime_blockers.append(
            "The checkpoint normalization does not include `transition_time`, so the export "
            "is missing the current variable-jump training contract."
        )
        recommendations.append(
            "Regenerate checkpoints after variable-jump training so `normalization.transition_time` "
            "is stored in the artifact."
        )

    if transition_time_name is not None and transition_time_name != TRANSITION_TIME_NAME:
        torch_runtime_blockers.append(
            f"The checkpoint transition-time feature is named `{transition_time_name}`, but the "
            f"current training contract expects `{TRANSITION_TIME_NAME}`."
        )
        recommendations.append(
            "Re-export the surrogate from a checkpoint produced by the current training code."
        )

    if conditioning_names != expected_conditioning_names:
        torch_runtime_blockers.append(
            "The checkpoint conditioning vector does not match the direct-jump contract: "
            f"expected {list(expected_conditioning_names)!r}, got {list(conditioning_names)!r}."
        )
        recommendations.append(
            "Export a checkpoint whose conditioning vector appends "
            f"`{TRANSITION_TIME_NAME}` after the physical parameters."
        )

    if not transition_days_conditioned:
        torch_runtime_blockers.append(
            "The export forward signature is fixed-step: it does not accept `transition_days`, "
            "so long horizons require recurrent rollout."
        )
        recommendations.append(
            "Train/export a direct-jump model with explicit `transition_days` conditioning."
        )

    if (not transition_days_conditioned) and rollout_steps > 10_000:
        torch_runtime_blockers.append(
            f"A {model_days:.1f}-day prediction would require about {rollout_steps:,} recurrent "
            "surrogate steps with the current fixed-step artifact."
        )
        recommendations.append(
            "Use log-spaced variable-jump training and export a long-horizon direct-jump artifact "
            "before attempting full nested-sampling retrieval."
        )

    if sampling_transition_days_min is None or sampling_transition_days_max is None:
        torch_runtime_blockers.append(
            "The checkpoint sampling metadata does not record a variable `transition_days` range, "
            "so there is no evidence this artifact was trained for flexible jumps from initial "
            "state toward equilibrium."
        )
        recommendations.append(
            "Train with explicit random forward-only jump sampling over a broad day range and "
            "store that range in checkpoint metadata."
        )

    if export_meta is None:
        recommendations.append(
            "Keep `model_export.meta.json` next to the TorchScript artifact so retrieval can "
            "validate the runtime contract without reloading the checkpoint."
        )
    else:
        meta_input = dict(export_meta.get("input", {}))
        if transition_days_conditioned and "transition_days" not in meta_input:
            torch_runtime_blockers.append(
                "The saved export metadata does not advertise `transition_days` even though "
                "the runtime signature suggests time conditioning."
            )
            recommendations.append(
                "Regenerate the export metadata from the same artifact so the contract is unambiguous."
            )

    if resolved_export.suffixes[-2:] != [".torchscript", ".pt"]:
        jax_runtime_blockers.append(
            "The retrieval runner currently knows how to load TorchScript exports only."
        )
    else:
        jax_runtime_blockers.append(
            "The checked-in export is TorchScript/PyTorch, so a JAX+jaxoplanet retrieval "
            "would need a framework bridge instead of staying inside one GPU-native graph."
        )
        recommendations.append(
            "Export a JAX-native surrogate, or retrain/export the surrogate in a JAX stack "
            "so the forward model and jaxoplanet share the same runtime."
        )

    direct_jump_torch_ready = not torch_runtime_blockers
    gpu_native_jax_ready = direct_jump_torch_ready and not jax_runtime_blockers

    return SurrogateArtifactContract(
        export_path=resolved_export,
        checkpoint_path=resolved_checkpoint,
        model_days=float(model_days),
        export_format="torchscript",
        forward_schema=str(export_model.forward.schema),
        export_arg_count=int(export_arg_count),
        input_shape=(int(shape["input_C"]), int(shape["H"]), int(shape["W"])),
        output_shape=(int(shape["target_C"]), int(shape["H"]), int(shape["W"])),
        param_names=param_names,
        conditioning_names=conditioning_names,
        input_fields=input_fields,
        output_fields=output_fields,
        transition_days_conditioned=bool(transition_days_conditioned),
        has_transition_time_stats=bool(has_transition_time_stats),
        transition_time_name=transition_time_name,
        sampling_transition_days_min=(
            None if sampling_transition_days_min is None else float(sampling_transition_days_min)
        ),
        sampling_transition_days_max=(
            None if sampling_transition_days_max is None else float(sampling_transition_days_max)
        ),
        fixed_step_days=float(fixed_step_days),
        rollout_steps_for_model_days=int(rollout_steps),
        direct_jump_torch_runtime_ready=bool(direct_jump_torch_ready),
        gpu_native_jax_retrieval_ready=bool(gpu_native_jax_ready),
        torch_runtime_blockers=tuple(torch_runtime_blockers),
        jax_runtime_blockers=tuple(jax_runtime_blockers),
        blockers=tuple(torch_runtime_blockers + jax_runtime_blockers),
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
    """Batched Torch inference wrapper for many-sample direct-jump retrieval calls."""

    def __init__(
        self,
        *,
        export_path: Path,
        checkpoint_path: Path,
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
        self.checkpoint: Dict[str, Any] = torch.load(
            self.contract.checkpoint_path,
            map_location="cpu",
        )
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

        shape = dict(self.checkpoint["shape"])
        self.input_shape = (int(shape["input_C"]), int(shape["H"]), int(shape["W"]))
        self.param_dim = int(len(self.checkpoint["param_names"]))

        # Example tensors come from checkpoint means rather than hard-coded
        # constants so smoke checks stay valid even when field transforms change.
        self._example_state_cpu = self._build_example_state()
        self._example_params_cpu = torch.tensor(
            self.checkpoint["normalization"]["params"]["mean"],
            dtype=torch.float32,
        ).reshape(1, -1)
        transition_mean_log10 = float(
            self.checkpoint["normalization"]["transition_time"]["mean"][0]
        )
        self._example_transition_days_cpu = torch.tensor(
            [10.0 ** transition_mean_log10],
            dtype=torch.float32,
        )

    def _build_example_state(self) -> torch.Tensor:
        """Construct one valid physical-space state from normalization statistics."""
        normalization = dict(self.checkpoint["normalization"])
        input_state = dict(normalization["input_state"])
        fields = list(self.checkpoint["input_fields"])
        zero_norm = torch.zeros((1, *self.input_shape), dtype=torch.float32)
        return _denormalize_state_with_stats(
            zero_norm,
            mean=torch.tensor(input_state["mean"], dtype=torch.float32),
            std=torch.tensor(input_state["std"], dtype=torch.float32),
            mode_codes=torch.tensor(
                _build_state_mode_codes(
                    fields=fields,
                    field_transforms=dict(input_state.get("field_transforms", {})),
                ),
                dtype=torch.int64,
            ),
            zscore_eps=float(input_state["zscore_eps"]),
            signed_log1p_scale=float(input_state["signed_log1p_scale"]),
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
                outputs.append(pred_batch.detach().cpu())

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
