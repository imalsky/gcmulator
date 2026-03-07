#!/usr/bin/env python3
"""Config-driven surrogate retrieval contract check and runtime smoke test.

Edit the config block near the top of this file. The default behavior is strict:
it writes a readiness report and then refuses to proceed unless the artifact is
compatible with the intended GPU-native JAX+jaxoplanet retrieval.

If you want to smoke-test the Torch direct-jump runtime for a valid artifact
before the JAX-native export exists, set ``require_gpu_native_jax`` to
``False``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from surrogate_backend import (
    SurrogateRuntimeConfig,
    TorchSurrogateRuntime,
    inspect_surrogate_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RetrievalRunConfig:
    """Top-level user-editable retrieval runner settings."""

    run_name: str = "v1"
    checkpoint_path: Path | None = None
    export_path: Path | None = None
    report_path: Path = (PROJECT_ROOT / "retrieval" / "artifact_readiness.json")
    model_days: float = 100.0

    # Keep this enabled when checking whether the artifact can satisfy the
    # intended JAX+jaxoplanet nested-sampling architecture.
    require_gpu_native_jax: bool = True

    # Smoke-test the batched Torch runtime when the artifact satisfies the
    # direct-jump contract, even if the full JAX-native path is still blocked.
    run_torch_smoke_test: bool = True
    smoke_batch_size: int = 32
    smoke_repeats: int = 10
    runtime: SurrogateRuntimeConfig = field(default_factory=SurrogateRuntimeConfig)


CONFIG = RetrievalRunConfig()


def _resolve_artifact_paths(config: RetrievalRunConfig) -> tuple[Path, Path]:
    """Resolve export/checkpoint paths from the config block."""
    if config.export_path is None or config.checkpoint_path is None:
        run_dir = (PROJECT_ROOT / "models" / str(config.run_name)).resolve()
        export_path = (run_dir / "model_export.torchscript.pt").resolve()
        checkpoint_path = (run_dir / "best.pt").resolve()
    else:
        export_path = Path(config.export_path).resolve()
        checkpoint_path = Path(config.checkpoint_path).resolve()
    return export_path, checkpoint_path


def main() -> None:
    """Write a contract report and optionally benchmark a valid Torch runtime."""
    export_path, checkpoint_path = _resolve_artifact_paths(CONFIG)
    contract = inspect_surrogate_artifact(
        export_path=export_path,
        checkpoint_path=checkpoint_path,
        model_days=float(CONFIG.model_days),
    )

    report: dict[str, object] = {
        "config": {
            **asdict(CONFIG),
            "checkpoint_path": str(checkpoint_path),
            "export_path": str(export_path),
            "report_path": str(Path(CONFIG.report_path).resolve()),
        },
        "artifact": contract.to_json_dict(),
    }

    if CONFIG.run_torch_smoke_test and contract.direct_jump_torch_runtime_ready:
        runtime = TorchSurrogateRuntime(
            export_path=export_path,
            checkpoint_path=checkpoint_path,
            runtime_config=CONFIG.runtime,
            model_days=float(CONFIG.model_days),
        )
        report["torch_smoke_test"] = runtime.benchmark(
            batch_size=int(CONFIG.smoke_batch_size),
            repeats=int(CONFIG.smoke_repeats),
        )

    report_path = Path(CONFIG.report_path).resolve()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not contract.direct_jump_torch_runtime_ready:
        joined = "\n".join(f"- {item}" for item in contract.torch_runtime_blockers)
        raise RuntimeError(
            "The current surrogate artifact is not runnable in the direct-jump Torch "
            "retrieval backend.\n"
            f"{joined}\n"
            f"Read {report_path} for the exact blocker list."
        )

    if CONFIG.require_gpu_native_jax and not contract.gpu_native_jax_retrieval_ready:
        joined = "\n".join(f"- {item}" for item in contract.jax_runtime_blockers)
        raise RuntimeError(
            "The current surrogate artifact is still not suitable for the intended "
            "GPU-native JAX+jaxoplanet retrieval.\n"
            f"{joined}\n"
            f"Read {report_path} for the exact blocker list."
        )


if __name__ == "__main__":
    main()
