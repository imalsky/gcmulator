#!/usr/bin/env python3
"""Config-driven surrogate retrieval contract check and runtime smoke test.

Edit the config block near the top of this file. The runner validates one
TorchScript export bundle, writes a readiness report, and optionally benchmarks
the optimized Torch runtime on either CPU or GPU.
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
    # Optional. When provided, the runner cross-checks the export metadata
    # against the original training checkpoint. The Torch runtime itself uses
    # the export bundle directly.
    checkpoint_path: Path | None = None
    export_path: Path | None = None
    report_path: Path = (PROJECT_ROOT / "retrieval" / "artifact_readiness.json")
    model_days: float = 100.0

    # Smoke-test the batched Torch runtime after the contract check succeeds.
    run_torch_smoke_test: bool = True
    smoke_batch_size: int = 32
    smoke_repeats: int = 10
    # Runtime settings only support explicit "cpu" or "gpu" choices for now.
    runtime: SurrogateRuntimeConfig = field(default_factory=SurrogateRuntimeConfig)


CONFIG = RetrievalRunConfig()


def _resolve_artifact_paths(config: RetrievalRunConfig) -> tuple[Path, Path | None]:
    """Resolve export/checkpoint paths from the config block."""
    if config.export_path is None:
        run_dir = (PROJECT_ROOT / "models" / str(config.run_name)).resolve()
        export_path = (run_dir / "model_export.torchscript.pt").resolve()
    else:
        export_path = Path(config.export_path).resolve()
    if config.checkpoint_path is None:
        if config.export_path is None:
            candidate_checkpoint = (
                PROJECT_ROOT / "models" / str(config.run_name) / "best.pt"
            ).resolve()
            checkpoint_path = (
                candidate_checkpoint if candidate_checkpoint.is_file() else None
            )
        else:
            checkpoint_path = None
    else:
        checkpoint_path = Path(config.checkpoint_path).resolve()
    return export_path, checkpoint_path


def _write_report(report_path: Path, payload: dict[str, object]) -> None:
    """Persist one JSON readiness report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Write a contract report and optionally benchmark a valid Torch runtime."""
    export_path, checkpoint_path = _resolve_artifact_paths(CONFIG)
    report_path = Path(CONFIG.report_path).resolve()
    base_report: dict[str, object] = {
        "config": {
            **asdict(CONFIG),
            "checkpoint_path": (
                None if checkpoint_path is None else str(checkpoint_path)
            ),
            "export_path": str(export_path),
            "report_path": str(report_path),
        }
    }
    try:
        contract = inspect_surrogate_artifact(
            export_path=export_path,
            checkpoint_path=checkpoint_path,
            model_days=float(CONFIG.model_days),
        )
    except Exception as exc:
        error_report = {
            **base_report,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        _write_report(report_path, error_report)
        print(json.dumps(error_report, indent=2))
        raise

    report: dict[str, object] = {
        **base_report,
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

    _write_report(report_path, report)
    print(json.dumps(report, indent=2))

    if not contract.direct_jump_torch_runtime_ready:
        joined = "\n".join(f"- {item}" for item in contract.torch_runtime_blockers)
        raise RuntimeError(
            "The current surrogate artifact is not runnable in the direct-jump Torch "
            "retrieval backend.\n"
            f"{joined}\n"
            f"Read {report_path} for the exact blocker list."
        )


if __name__ == "__main__":
    main()
