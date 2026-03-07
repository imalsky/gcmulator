#!/usr/bin/env python3
"""Fail-fast entrypoint for surrogate retrieval readiness.

The current checked-in artifact is not suitable for the requested fast,
GPU-native JAX+jaxoplanet retrieval. This script makes that contract explicit
instead of pretending the retrieval is runnable.
"""

from __future__ import annotations

import json
from pathlib import Path

from surrogate_backend import inspect_surrogate_artifact


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_PATH = PROJECT_ROOT / "models" / "v1" / "model_export.torchscript.pt"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "v1" / "best.pt"
MODEL_DAYS = 100.0
REPORT_PATH = PROJECT_ROOT / "retrieval" / "artifact_readiness.json"


def main() -> None:
    contract = inspect_surrogate_artifact(
        export_path=EXPORT_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        model_days=MODEL_DAYS,
    )
    REPORT_PATH.write_text(json.dumps(contract.to_json_dict(), indent=2), encoding="utf-8")

    print(json.dumps(contract.to_json_dict(), indent=2))
    if not contract.gpu_native_jax_retrieval_ready:
        joined = "\n".join(f"- {item}" for item in contract.blockers)
        raise RuntimeError(
            "The current surrogate artifact is not ready for a fast GPU-native "
            "JAX+jaxoplanet retrieval.\n"
            f"{joined}\n"
            f"Read {REPORT_PATH} and the retrieval README for the required artifact changes."
        )


if __name__ == "__main__":
    main()
