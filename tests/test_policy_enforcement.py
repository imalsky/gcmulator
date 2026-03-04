from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from src.constants import (
    LOG10_INVERSE_CLIP_MAX,
    LOG10_INVERSE_CLIP_MIN,
    PARAM_NORM_CLIP_ABS,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    STD_FLOOR,
)
from src.config import load_config
from src.config import time_days_to_rollout_steps
from src.my_swamp_backend import enforce_no_tpu_backend
from src.training import preprocess_dataset


def _literal_value(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _literal_value(node.operand)
        if val is not None:
            return -val
    return None


def _find_banned_numeric_literals(path: Path, banned_values: set[float]) -> list[tuple[int, float]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[tuple[int, float]] = []
    for node in ast.walk(tree):
        value = _literal_value(node)
        if value is not None and value in banned_values:
            lineno = getattr(node, "lineno", 0)
            hits.append((int(lineno), value))
    return sorted(hits)


def test_time_mapping_fail_fast() -> None:
    assert time_days_to_rollout_steps(100.0, default_time_days=100.0, rollout_steps_at_default_time=16) == 16
    with pytest.raises(ValueError, match="time_days must be > 0"):
        time_days_to_rollout_steps(0.0, default_time_days=100.0, rollout_steps_at_default_time=16)
    with pytest.raises(ValueError, match="time_days must be finite"):
        time_days_to_rollout_steps(float("nan"), default_time_days=100.0, rollout_steps_at_default_time=16)
    with pytest.raises(ValueError, match="default_time_days must be > 0"):
        time_days_to_rollout_steps(1.0, default_time_days=0.0, rollout_steps_at_default_time=16)
    with pytest.raises(ValueError, match="default_time_days must be finite"):
        time_days_to_rollout_steps(1.0, default_time_days=float("inf"), rollout_steps_at_default_time=16)
    with pytest.raises(ValueError, match="rollout_steps_at_default_time must be >="):
        time_days_to_rollout_steps(1.0, default_time_days=100.0, rollout_steps_at_default_time=0)


def test_preprocess_cached_meta_fail_fast(tmp_path: Path) -> None:
    cfg_text = """
{
  "paths": {
    "dataset_dir": "raw",
    "processed_dir": "processed",
    "model_dir": "model",
    "overwrite_processed": false
  },
  "solver": {
    "M": 42,
    "dt_seconds": 60.0,
    "default_time_days": 100.0,
    "starttime_index": 2
  },
  "sampling": {
    "seed": 1,
    "n_sims": 8,
    "parameters": [
      {"name": "a_m", "dist": "fixed", "value": 8.2e7},
      {"name": "omega_rad_s", "dist": "fixed", "value": 3.2e-5},
      {"name": "Phibar", "dist": "fixed", "value": 3.0e5},
      {"name": "DPhieq", "dist": "fixed", "value": 1.0e6},
      {"name": "taurad_hours", "dist": "uniform", "min": 1.0, "max": 30.0},
      {"name": "taudrag_hours", "dist": "uniform", "min": 1.0, "max": 30.0},
      {"name": "g_m_s2", "dist": "fixed", "value": 9.8},
      {"name": "K6", "dist": "fixed", "value": 1.24e33},
      {"name": "K6Phi", "dist": "fixed", "value": 0.0}
    ]
  }
}
"""
    config_path = tmp_path / "config.json"
    config_path.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(config_path)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "processed_meta.json").write_text('{"fields": ["Phi"]}', encoding="utf-8")

    with pytest.raises(KeyError, match="missing required keys"):
        preprocess_dataset(cfg, config_path=config_path)


def test_no_tpu_backend_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setenv("JAX_PLATFORM_NAME", "tpu")
    enforce_no_tpu_backend()
    assert "JAX_PLATFORMS" not in os.environ
    assert "JAX_PLATFORM_NAME" not in os.environ

    monkeypatch.setenv("JAX_PLATFORMS", "tpu,gpu,cpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    enforce_no_tpu_backend()
    assert "tpu" not in os.environ["JAX_PLATFORMS"].lower()
    assert os.environ["JAX_PLATFORMS"] == "gpu,cpu"


def test_policy_constants_expected_values() -> None:
    assert SECONDS_PER_HOUR == 3600.0
    assert SECONDS_PER_DAY == 86400.0
    assert STD_FLOOR == 1.0e-12
    assert LOG10_INVERSE_CLIP_MIN == -30.0
    assert LOG10_INVERSE_CLIP_MAX == 30.0
    assert PARAM_NORM_CLIP_ABS == 1.0e6


def test_no_banned_magic_literals_in_core_modules() -> None:
    root = Path(__file__).resolve().parents[1] / "src"
    files = [
        root / "config.py",
        root / "sampling.py",
        root / "my_swamp_backend.py",
        root / "normalization.py",
        root / "data_generation.py",
        root / "training.py",
        root / "modeling.py",
    ]
    banned_values = {
        float(SECONDS_PER_HOUR),
        float(SECONDS_PER_DAY),
        float(STD_FLOOR),
        float(PARAM_NORM_CLIP_ABS),
        float(LOG10_INVERSE_CLIP_MIN),
        float(LOG10_INVERSE_CLIP_MAX),
    }

    findings: list[str] = []
    for file_path in files:
        hits = _find_banned_numeric_literals(file_path, banned_values)
        for line, value in hits:
            findings.append(f"{file_path.name}:{line}:{value}")

    assert not findings, "Found banned magic numeric literals in core modules: " + ", ".join(findings)


def test_dead_code_regressions_do_not_return() -> None:
    root = Path(__file__).resolve().parents[1] / "src"
    proj = Path(__file__).resolve().parents[1]

    assert not (root / "retrieval_adapter.py").exists()
    assert not (root / "inference.py").exists()
    assert not (root / "cli.py").exists()
    assert not (root / "types.py").exists()
    assert not (root / "time_mapping.py").exists()
    assert not (proj / "main.py").exists()
    assert not (proj / "create_dataset.py").exists()
    assert not (proj / "create_emulator.py").exists()
    assert not (proj / "inference.py").exists()

    sampling_text = (root / "sampling.py").read_text(encoding="utf-8")
    normalization_text = (root / "normalization.py").read_text(encoding="utf-8")

    assert "def extended9_param_vector" not in sampling_text
    assert "def denormalize_params" not in normalization_text
    assert "def per_channel_rmse" not in normalization_text
