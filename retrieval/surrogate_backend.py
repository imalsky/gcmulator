"""Artifact inspection helpers for GPU-native surrogate retrieval readiness."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass(frozen=True)
class SurrogateArtifactContract:
    """Summarize whether a saved surrogate can support fast GPU-native retrieval."""

    export_path: Path
    checkpoint_path: Path
    export_format: str
    forward_schema: str
    export_arg_count: int
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
    rollout_steps_for_100_days: int
    gpu_native_jax_retrieval_ready: bool
    blockers: tuple[str, ...]
    recommendations: tuple[str, ...]

    def to_json_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly payload."""
        return {
            "export_path": str(self.export_path),
            "checkpoint_path": str(self.checkpoint_path),
            "export_format": self.export_format,
            "forward_schema": self.forward_schema,
            "export_arg_count": int(self.export_arg_count),
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
            "rollout_steps_for_100_days": int(self.rollout_steps_for_100_days),
            "gpu_native_jax_retrieval_ready": bool(self.gpu_native_jax_retrieval_ready),
            "blockers": list(self.blockers),
            "recommendations": list(self.recommendations),
        }


def _load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_surrogate_artifact(
    *,
    export_path: Path,
    checkpoint_path: Path,
    model_days: float = 100.0,
) -> SurrogateArtifactContract:
    """Inspect the current surrogate artifact against retrieval requirements.

    This intentionally answers a narrow question:
    can the saved artifact support a fast, GPU-native retrieval through a JAX /
    jaxoplanet pipeline without a host callback and without fixed-step rollout?
    """
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
    meta_name = resolved_export.name.replace(".torchscript.pt", ".meta.json")
    export_meta = _load_optional_json(resolved_export.with_name(meta_name))

    solver = dict(checkpoint["solver"])
    sampling = dict(checkpoint.get("sampling", {}))
    normalization = dict(checkpoint.get("normalization", {}))
    input_fields = tuple(str(value) for value in checkpoint["input_fields"])
    output_fields = tuple(str(value) for value in checkpoint["target_fields"])
    param_names = tuple(str(value) for value in checkpoint["param_names"])
    conditioning_names = tuple(str(value) for value in checkpoint.get("conditioning_names", ()))

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

    blockers: List[str] = []
    recommendations: List[str] = []

    blockers.append(
        "The checked-in export is TorchScript/PyTorch, so a JAX+jaxoplanet retrieval "
        "would need a framework bridge instead of staying inside one GPU-native graph."
    )
    recommendations.append(
        "Export a JAX-native surrogate, or retrain/export the surrogate in a JAX stack "
        "so the forward model and jaxoplanet share the same runtime."
    )

    if not has_transition_time_stats:
        blockers.append(
            "The checkpoint normalization does not include `transition_time`, so the export "
            "is missing the current variable-jump training contract."
        )
        recommendations.append(
            "Regenerate checkpoints after variable-jump training so `normalization.transition_time` "
            "is stored in the artifact."
        )

    expected_time_name = "log10_transition_days"
    if transition_time_name is not None and transition_time_name != expected_time_name:
        blockers.append(
            f"The checkpoint transition-time feature is named `{transition_time_name}`, but the "
            f"current training contract expects `{expected_time_name}`."
        )
        recommendations.append(
            "Re-export the surrogate from a checkpoint produced by the current training code."
        )

    if conditioning_names and expected_time_name not in conditioning_names:
        blockers.append(
            f"The checkpoint conditioning vector is {list(conditioning_names)}, which does not "
            f"include `{expected_time_name}`."
        )
        recommendations.append(
            "Export a checkpoint whose conditioning vector includes the explicit time feature."
        )

    if not transition_days_conditioned:
        blockers.append(
            "The export forward signature is fixed-step: it does not accept `transition_days`, "
            "so long horizons require recurrent rollout."
        )
        recommendations.append(
            "Train/export a direct-jump model with explicit `transition_days` conditioning."
        )

    if rollout_steps > 10_000:
        blockers.append(
            f"A {model_days:.1f}-day prediction would require about {rollout_steps:,} recurrent "
            "surrogate steps with the current fixed-step artifact."
        )
        recommendations.append(
            "Use log-spaced variable-jump training and export a long-horizon direct-jump artifact "
            "before attempting full nested-sampling retrieval."
        )

    if sampling_transition_days_min is None or sampling_transition_days_max is None:
        blockers.append(
            "The checkpoint sampling metadata does not record a variable `transition_days` range, "
            "so there is no evidence this artifact was trained for flexible jumps from initial "
            "state toward equilibrium."
        )
        recommendations.append(
            "Train with explicit random forward-only jump sampling over a broad day range and "
            "store that range in checkpoint metadata."
        )

    if export_meta is not None:
        meta_input = dict(export_meta.get("input", {}))
        if transition_days_conditioned and "transition_days" not in meta_input:
            blockers.append(
                "The saved export metadata does not advertise `transition_days` even though "
                "the runtime signature suggests time conditioning."
            )
            recommendations.append(
                "Regenerate the export metadata from the same artifact so the contract is unambiguous."
            )

    gpu_native_ready = not blockers

    return SurrogateArtifactContract(
        export_path=resolved_export,
        checkpoint_path=resolved_checkpoint,
        export_format="torchscript",
        forward_schema=str(export_model.forward.schema),
        export_arg_count=int(export_arg_count),
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
        rollout_steps_for_100_days=int(rollout_steps),
        gpu_native_jax_retrieval_ready=bool(gpu_native_ready),
        blockers=tuple(blockers),
        recommendations=tuple(recommendations),
    )
