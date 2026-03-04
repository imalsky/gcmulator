from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_config
from src.training import train_emulator


@pytest.mark.smoke
def test_train_smoke_with_synthetic_data(tmp_path: Path) -> None:
    pytest.importorskip("torch_harmonics")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    field_names = ["Phi", "U", "V", "eta", "delta"]
    param_names = ["a_m", "omega_rad_s", "Phibar", "DPhieq", "taurad_s", "taudrag_s", "g_m_s2", "K6", "K6Phi"]

    rng = np.random.default_rng(0)
    for i in range(8):
        state = rng.normal(size=(5, 8, 16)).astype(np.float32)
        # make Phi positive-ish to support optional log transforms
        state[0] = np.abs(state[0]) + 1.0

        params = np.array(
            [
                8.2e7,
                3.2e-5,
                3.0e5,
                1.0e6,
                10.0 * 3600.0,
                6.0 * 3600.0,
                9.8,
                1.24e33,
                0.0 if i % 2 == 0 else 1.0e32,
            ],
            dtype=np.float64,
        )

        np.savez_compressed(
            raw_dir / f"sim_{i:06d}.npz",
            state_final=state,
            fields=np.asarray(field_names, dtype=object),
            params=params,
            param_names=np.asarray(param_names, dtype=object),
            time_days=np.asarray(100.0, dtype=np.float64),
            dt_seconds=np.asarray(60.0, dtype=np.float64),
            M=np.asarray(42, dtype=np.int64),
            nlat=np.asarray(8, dtype=np.int64),
            nlon=np.asarray(16, dtype=np.int64),
            lat_order=np.asarray("north_to_south", dtype=object),
            lon_origin=np.asarray("0_to_2pi", dtype=object),
            lon_shift=np.asarray(-8, dtype=np.int64),
        )

    cfg_dict = {
        "paths": {
            "dataset_dir": "raw",
            "processed_dir": "processed",
            "model_dir": "model",
            "overwrite_processed": True,
        },
        "solver": {
            "M": 42,
            "dt_seconds": 60.0,
            "default_time_days": 100.0,
            "starttime_index": 2,
        },
        "sampling": {
            "seed": 42,
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
                {"name": "K6Phi", "dist": "fixed", "value": 0.0},
            ],
        },
        "model": {
            "grid": "legendre-gauss",
            "grid_internal": "legendre-gauss",
            "embed_dim": 8,
            "num_layers": 1,
            "scale_factor": 1,
            "rollout_steps_at_default_time": 2,
            "include_coord_channels": True,
            "include_param_maps": True,
            "ic": {
                "hidden_dim": 16,
                "num_layers": 1,
                "rand_basis_count": 1,
                "basis": ["const", "sin_lat", "cos_lat", "sin_lon", "cos_lon"],
            },
        },
        "training": {
            "device": "cpu",
            "amp_mode": "none",
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "val_fraction": 0.25,
            "split_seed": 0,
            "learning_rate": 1e-3,
            "grad_clip_norm": 1.0,
            "scheduler": {"type": "none"},
        },
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    cfg = load_config(config_path)
    summary = train_emulator(cfg, config_path=config_path)

    best_ckpt = Path(summary["best_checkpoint"])
    assert best_ckpt.exists()
    assert Path(summary["history_path"]).exists()
    assert Path(summary["processed_meta"]).exists()
    assert Path(summary["resolved_config_path"]).exists()
    assert Path(summary["original_config_copy_path"]).exists()

    ckpt = torch.load(best_ckpt, map_location="cpu")
    assert "resolved_config" in ckpt
    assert "source_config_path" in ckpt
    assert Path(ckpt["source_config_path"]).resolve() == config_path.resolve()


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("GCMULATOR_RUN_SLOW", "0") != "1", reason="Set GCMULATOR_RUN_SLOW=1 to run MY_SWAMP generation smoke")
def test_my_swamp_generation_smoke(tmp_path: Path) -> None:
    pytest.importorskip("my_swamp")

    from src.data_generation import generate_dataset

    cfg_dict = {
        "paths": {
            "dataset_dir": "raw",
            "processed_dir": "processed",
            "model_dir": "model",
            "overwrite_dataset": True,
        },
        "solver": {
            "M": 42,
            "dt_seconds": 60.0,
            "default_time_days": 1.0,
            "starttime_index": 2,
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
                {"name": "K6Phi", "dist": "fixed", "value": 0.0},
            ],
        },
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    cfg = load_config(config_path)

    manifest = generate_dataset(cfg, config_path=config_path)
    assert manifest["n_sims_written"] == 8
    assert (tmp_path / "raw" / "manifest.json").exists()
