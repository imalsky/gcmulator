"""Visualize one held-out multi-jump prognostic prediction from the test split."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_MPL_CACHE_DIR = PROJECT_CACHE_DIR / "mplconfig"
PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_DIR.resolve()))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1")
MPL_CACHE_DIR = Path(os.environ.get("GCMULATOR_MPLCONFIGDIR", str(DEFAULT_MPL_CACHE_DIR))).resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch


SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling import build_state_conditioned_transition_model, ensure_torch_harmonics_importable
from config import resolve_path
from my_swamp_backend import (
    diagnose_winds,
    ensure_my_swamp_importable,
    params_to_conditioning_vector,
    reconstruct_full_state_from_prognostics,
    run_trajectory_window,
)
from normalization import (
    denormalize_state_tensor,
    normalize_conditioning,
    normalize_state_tensor,
    stats_from_json,
)
from sampling import to_extended9


FIGURE_DPI = 180
DEFAULT_FIGURE_NAME = "test_rollout_true_vs_pred.png"
STYLE_PATH = Path(__file__).resolve().with_name("science.mplstyle")
PHI_CHANNEL_INDEX = 0
FIELD_NAME = "Phi"
COLOR_MAP = "Blues"
QUIVER_STRIDE = 8
QUIVER_COLOR = "#08306b"

# User-editable run settings
RUN_NAME = "v1"
RUN_DIR: Path | None = (PROJECT_ROOT / "models" / RUN_NAME).resolve()
CHECKPOINT_PATH: Path | None = None
PROCESSED_DIR: Path | None = None
TEST_SHARD_INDEX = 0
INPUT_TIME_DAYS = 0.0
TARGET_TIME_DAYS = 10.0
NUM_JUMPS = 1
DEVICE_MODE = "auto"
FIGURE_PATH: Path | None = None


def _dict_to_namespace(obj: Any) -> Any:
    """Convert nested dict/list structures into namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: _dict_to_namespace(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(value) for value in obj]
    return obj


def _resolve_checkpoint_path(*, run_dir: Path | None, checkpoint: Path | None) -> Path:
    """Resolve checkpoint path from top-level run settings."""
    if checkpoint is not None:
        resolved = checkpoint.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved
    if run_dir is None:
        raise ValueError("Set RUN_DIR or CHECKPOINT_PATH at the top of this file")
    ckpt_path = (run_dir.resolve() / "best.pt").resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def _resolve_device(mode: str) -> torch.device:
    """Resolve inference device."""
    normalized = str(mode).lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested DEVICE_MODE='gpu' but CUDA is unavailable")
        return torch.device("cuda")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device mode: {mode}")


def _resolve_processed_dir(ckpt: Dict[str, Any]) -> Path:
    """Resolve the processed directory, defaulting to the checkpoint's dataset."""
    if PROCESSED_DIR is not None:
        resolved = PROCESSED_DIR.resolve()
    else:
        source_config_path = ckpt.get("source_config_path")
        resolved_config = ckpt.get("resolved_config")
        if not isinstance(source_config_path, str) or not isinstance(resolved_config, dict):
            raise ValueError(
                "Checkpoint is missing source config metadata; set PROCESSED_DIR explicitly"
            )
        paths_cfg = resolved_config.get("paths")
        if not isinstance(paths_cfg, dict):
            raise ValueError(
                "Checkpoint is missing resolved paths metadata; set PROCESSED_DIR explicitly"
            )
        processed_dir_value = paths_cfg.get("processed_dir")
        if not isinstance(processed_dir_value, str):
            raise ValueError(
                "Checkpoint is missing paths.processed_dir; set PROCESSED_DIR explicitly"
            )
        resolved = resolve_path(Path(source_config_path), processed_dir_value)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Processed directory not found: {resolved}")
    return resolved


def _denormalize_params(params_norm: np.ndarray, stats: Any) -> Dict[str, float]:
    """Invert parameter normalization back to physical values."""
    params_norm = np.asarray(params_norm, dtype=np.float64)
    params_phys = params_norm * (stats.std + stats.zscore_eps) + stats.mean
    const_mask = np.asarray(stats.is_constant, dtype=bool)
    if np.any(const_mask):
        params_phys[const_mask] = stats.mean[const_mask]
    return {
        str(name): float(params_phys[index])
        for index, name in enumerate(stats.param_names)
    }


def _apply_plot_style() -> None:
    """Load the shared plotting style."""
    if not STYLE_PATH.is_file():
        raise FileNotFoundError(f"Plot style not found: {STYLE_PATH}")
    plt.style.use(str(STYLE_PATH))
    plt.rcParams["savefig.dpi"] = int(FIGURE_DPI)


def _color_limits(true_field: np.ndarray, pred_field: np.ndarray) -> tuple[float, float]:
    """Compute shared robust color limits for one field."""
    values = np.concatenate([true_field.reshape(-1), pred_field.reshape(-1)]).astype(np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.quantile(values, 0.01))
    vmax = float(np.quantile(values, 0.99))
    if not np.isfinite(vmin):
        vmin = float(np.min(values))
    if not np.isfinite(vmax):
        vmax = float(np.max(values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _add_quiver(ax: Any, u_field: np.ndarray, v_field: np.ndarray) -> None:
    """Overlay a sparse wind quiver on one axis."""
    y_idx = np.arange(0, int(u_field.shape[0]), int(QUIVER_STRIDE))
    x_idx = np.arange(0, int(u_field.shape[1]), int(QUIVER_STRIDE))
    x_grid, y_grid = np.meshgrid(x_idx, y_idx)
    u_sub = np.asarray(u_field[np.ix_(y_idx, x_idx)], dtype=np.float64)
    v_sub = np.asarray(v_field[np.ix_(y_idx, x_idx)], dtype=np.float64)
    ax.quiver(
        x_grid,
        y_grid,
        u_sub,
        v_sub,
        color=QUIVER_COLOR,
        pivot="mid",
        alpha=0.8,
        width=0.0022,
    )


def _save_figure(
    *,
    true_state: np.ndarray,
    pred_state: np.ndarray,
    true_winds: tuple[np.ndarray, np.ndarray],
    pred_winds: tuple[np.ndarray, np.ndarray],
    shard_name: str,
    context_label: str,
    input_day: float,
    target_day: float,
    out_path: Path,
) -> None:
    """Save a 1x2 Phi comparison figure."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.5),
        dpi=int(FIGURE_DPI),
        constrained_layout=True,
    )
    true_field = np.asarray(true_state[PHI_CHANNEL_INDEX], dtype=np.float64)
    pred_field = np.asarray(pred_state[PHI_CHANNEL_INDEX], dtype=np.float64)
    vmin, vmax = _color_limits(true_field, pred_field)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im_true = axes[0].imshow(
        true_field,
        origin="lower",
        cmap=COLOR_MAP,
        norm=norm,
        interpolation="bicubic",
        aspect="auto",
    )
    axes[1].imshow(
        pred_field,
        origin="lower",
        cmap=COLOR_MAP,
        norm=norm,
        interpolation="bicubic",
        aspect="auto",
    )
    axes[0].set_title(f"True {FIELD_NAME}")
    axes[1].set_title(f"Predicted {FIELD_NAME}")
    axes[0].set_xlabel("Longitude Index")
    axes[1].set_xlabel("Longitude Index")
    axes[0].set_ylabel("Latitude Index")
    axes[1].set_ylabel("Latitude Index")
    _add_quiver(axes[0], true_winds[0], true_winds[1])
    _add_quiver(axes[1], pred_winds[0], pred_winds[1])
    fig.colorbar(im_true, ax=axes, shrink=0.9)

    rmse = float(np.sqrt(np.mean((pred_field - true_field) ** 2)))
    fig.suptitle(
        f"{shard_name} | {context_label} | input_day={input_day:.6f} | "
        f"target_day={target_day:.6f} | {FIELD_NAME} RMSE={rmse:.3e}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _load_test_split_entries(processed_dir: Path) -> list[dict[str, Any]]:
    """Return held-out processed shard entries from the test split."""
    meta_path = (processed_dir / "processed_meta.json").resolve()
    if not meta_path.is_file():
        raise FileNotFoundError(f"Processed metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    split_entries = list(meta["splits"]["test"])
    if not split_entries:
        raise RuntimeError("Test split is empty")
    return split_entries


def _resolve_time_steps(
    *,
    input_time_days: float,
    target_time_days: float,
    dt_seconds: float,
    max_time_days: float,
) -> tuple[int, int, float, float]:
    """Convert requested day-valued times to validated solver steps."""
    if input_time_days < 0.0:
        raise ValueError("INPUT_TIME_DAYS must be >= 0")
    if target_time_days <= input_time_days:
        raise ValueError("TARGET_TIME_DAYS must be > INPUT_TIME_DAYS")
    if target_time_days > max_time_days:
        raise ValueError(
            "TARGET_TIME_DAYS exceeds the simulated horizon from the checkpoint: "
            f"{target_time_days} > {max_time_days}"
        )

    input_step = int(round(float(input_time_days) * 86400.0 / float(dt_seconds)))
    target_step = int(round(float(target_time_days) * 86400.0 / float(dt_seconds)))
    if target_step <= input_step:
        raise ValueError("Resolved target step must be greater than the input step")

    actual_input_day = float(input_step) * float(dt_seconds) / 86400.0
    actual_target_day = float(target_step) * float(dt_seconds) / 86400.0
    return input_step, target_step, actual_input_day, actual_target_day


def _build_jump_schedule(*, input_step: int, target_step: int, num_jumps: int) -> np.ndarray:
    """Split one solver-step interval into ``num_jumps`` positive integer jumps."""
    if num_jumps < 1:
        raise ValueError("NUM_JUMPS must be >= 1")
    total_steps = int(target_step) - int(input_step)
    if total_steps < 1:
        raise ValueError("target_step must be greater than input_step")
    if int(num_jumps) > total_steps:
        raise ValueError(
            "NUM_JUMPS cannot exceed the number of solver steps between input and target"
        )

    base = total_steps // int(num_jumps)
    remainder = total_steps % int(num_jumps)
    jump_steps = np.full((int(num_jumps),), base, dtype=np.int64)
    jump_steps[:remainder] += 1
    if np.any(jump_steps < 1):
        raise RuntimeError("Jump schedule construction produced a non-positive jump")
    return jump_steps


def _load_test_rollout_case(
    *,
    processed_dir: Path,
    ckpt: Dict[str, Any],
    stats: Any,
    test_shard_index: int,
    input_time_days: float,
    target_time_days: float,
    num_jumps: int,
) -> tuple[Any, np.ndarray, np.ndarray, str, float, float, np.ndarray]:
    """Load one held-out test simulation and construct one rollout case."""
    test_entries = _load_test_split_entries(processed_dir)
    if test_shard_index < 0 or test_shard_index >= len(test_entries):
        raise IndexError(
            f"TEST_SHARD_INDEX={test_shard_index} is out of range for {len(test_entries)} test shards"
        )

    shard_entry = dict(test_entries[test_shard_index])
    shard_name = str(shard_entry["file"])
    shard_path = (processed_dir / shard_name).resolve()
    if not shard_path.is_file():
        raise FileNotFoundError(f"Processed shard not found: {shard_path}")

    with np.load(shard_path, allow_pickle=False) as npz:
        params_norm = np.asarray(npz["params_norm"], dtype=np.float32)

    params = to_extended9(_denormalize_params(params_norm, stats.params))
    solver_cfg = dict(ckpt["solver"])
    input_step, target_step, actual_input_day, actual_target_day = _resolve_time_steps(
        input_time_days=float(input_time_days),
        target_time_days=float(target_time_days),
        dt_seconds=float(solver_cfg["dt_seconds"]),
        max_time_days=float(solver_cfg["default_time_days"]),
    )
    jump_schedule = _build_jump_schedule(
        input_step=input_step,
        target_step=target_step,
        num_jumps=int(num_jumps),
    )

    state_inputs_phys, state_targets_phys = run_trajectory_window(
        params,
        M=int(solver_cfg["M"]),
        dt_seconds=float(solver_cfg["dt_seconds"]),
        time_days=float(solver_cfg["default_time_days"]),
        starttime_index=int(solver_cfg["starttime_index"]),
        anchor_steps=np.asarray([input_step], dtype=np.int64),
        target_steps=np.asarray([target_step], dtype=np.int64),
    )
    return (
        params,
        np.asarray(state_inputs_phys[0], dtype=np.float32),
        np.asarray(state_targets_phys[0], dtype=np.float32),
        shard_name,
        actual_input_day,
        actual_target_day,
        jump_schedule,
    )


def _predict_multi_jump(
    *,
    model: torch.nn.Module,
    stats: Any,
    params: Any,
    initial_state_phys: np.ndarray,
    jump_schedule: Sequence[int],
    solver_cfg: Dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Run one recursive multi-jump rollout and return the final prognostic target."""
    current_state_phys = np.asarray(initial_state_phys, dtype=np.float32)
    params_vector = params_to_conditioning_vector(params)

    for jump_index, jump_steps in enumerate(jump_schedule):
        jump_days = float(jump_steps) * float(solver_cfg["dt_seconds"]) / 86400.0
        current_state_norm = normalize_state_tensor(
            current_state_phys[None, ...],
            stats.input_state,
        ).astype(np.float32)
        conditioning_norm = normalize_conditioning(
            params_vector,
            np.asarray([jump_days], dtype=np.float64),
            param_stats=stats.params,
            transition_time_stats=stats.transition_time,
        ).astype(np.float32)

        with torch.inference_mode():
            pred_norm = model(
                torch.from_numpy(current_state_norm).to(device=device),
                torch.from_numpy(conditioning_norm).to(device=device),
            )
        pred_phys = denormalize_state_tensor(
            pred_norm.detach().cpu().numpy(),
            stats.target_state,
        )[0]
        if jump_index == (len(jump_schedule) - 1):
            return np.asarray(pred_phys, dtype=np.float32)

        current_state_phys = reconstruct_full_state_from_prognostics(
            np.asarray(pred_phys, dtype=np.float64),
            params=params,
            M=int(solver_cfg["M"]),
            dt_seconds=float(solver_cfg["dt_seconds"]),
        ).astype(np.float32)

    raise RuntimeError("Multi-jump rollout did not produce a final prediction")


def _diagnose_target_winds(
    target_state: np.ndarray,
    *,
    target_field_names: Sequence[str],
    params: Any,
    solver_cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Diagnose physical-space winds from prognostic target channels."""
    field_index = {str(name): idx for idx, name in enumerate(target_field_names)}
    return diagnose_winds(
        np.asarray(target_state[field_index["eta"]], dtype=np.float64),
        np.asarray(target_state[field_index["delta"]], dtype=np.float64),
        params=params,
        M=int(solver_cfg["M"]),
        dt_seconds=float(solver_cfg["dt_seconds"]),
    )


def main() -> None:
    """Load one held-out test case, run a recursive rollout, and save a prognostic plot."""
    _apply_plot_style()
    ckpt_path = _resolve_checkpoint_path(run_dir=RUN_DIR, checkpoint=CHECKPOINT_PATH)
    run_dir = ckpt_path.parent
    figure_path = (
        FIGURE_PATH.resolve()
        if FIGURE_PATH is not None
        else (run_dir / "plots" / DEFAULT_FIGURE_NAME).resolve()
    )

    device = _resolve_device(DEVICE_MODE)
    ensure_torch_harmonics_importable()
    ensure_my_swamp_importable()
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=device)
    stats = stats_from_json(ckpt["normalization"])
    processed_dir = _resolve_processed_dir(ckpt)
    (
        params,
        initial_state_phys,
        true_target_phys,
        shard_name,
        actual_input_day,
        actual_target_day,
        jump_schedule,
    ) = _load_test_rollout_case(
        processed_dir=processed_dir,
        ckpt=ckpt,
        stats=stats,
        test_shard_index=int(TEST_SHARD_INDEX),
        input_time_days=float(INPUT_TIME_DAYS),
        target_time_days=float(TARGET_TIME_DAYS),
        num_jumps=int(NUM_JUMPS),
    )

    model_cfg = _dict_to_namespace(ckpt["model_config"])
    shape = dict(ckpt["shape"])
    input_fields = list(ckpt["input_fields"])
    target_fields = list(ckpt["target_fields"])
    residual_input_indices = [input_fields.index(field_name) for field_name in target_fields]
    model = build_state_conditioned_transition_model(
        img_size=(int(shape["H"]), int(shape["W"])),
        input_state_chans=int(shape["input_C"]),
        target_state_chans=int(shape["target_C"]),
        param_dim=int(len(ckpt["conditioning_names"])),
        residual_input_indices=residual_input_indices,
        cfg_model=model_cfg,
        lat_order="north_to_south",
        lon_origin="0_to_2pi",
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device=device).eval()

    pred_target_phys = _predict_multi_jump(
        model=model,
        stats=stats,
        params=params,
        initial_state_phys=initial_state_phys,
        jump_schedule=jump_schedule,
        solver_cfg=dict(ckpt["solver"]),
        device=device,
    )
    true_winds = _diagnose_target_winds(
        true_target_phys,
        target_field_names=stats.target_state.field_names,
        params=params,
        solver_cfg=dict(ckpt["solver"]),
    )
    pred_winds = _diagnose_target_winds(
        pred_target_phys,
        target_field_names=stats.target_state.field_names,
        params=params,
        solver_cfg=dict(ckpt["solver"]),
    )

    mean_jump_days = float(np.mean(np.asarray(jump_schedule, dtype=np.float64))) * float(
        ckpt["solver"]["dt_seconds"]
    ) / 86400.0
    _save_figure(
        true_state=true_target_phys,
        pred_state=pred_target_phys,
        true_winds=true_winds,
        pred_winds=pred_winds,
        shard_name=shard_name,
        context_label=f"test_shard={TEST_SHARD_INDEX} | jumps={len(jump_schedule)} | mean_jump_days={mean_jump_days:.6f}",
        input_day=actual_input_day,
        target_day=actual_target_day,
        out_path=figure_path,
    )
    print(f"Saved prognostic prediction figure: {figure_path}")


if __name__ == "__main__":
    main()
