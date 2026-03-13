"""Visualize one held-out prognostic prediction from the test split."""

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

from gcmulator.modeling import (
    build_state_conditioned_transition_model,
    ensure_torch_harmonics_importable,
)
from gcmulator.my_swamp_backend import (
    diagnose_winds,
    params_to_conditioning_vector,
)
from gcmulator.normalization import (
    denormalize_state_tensor,
    normalize_conditioning,
    normalize_state_tensor,
    stats_from_json,
)
from gcmulator.sampling import to_extended9


FIGURE_DPI = 180
DEFAULT_FIGURE_NAME = "test_direct_jump_true_vs_pred.png"
STYLE_PATH = Path(__file__).resolve().with_name("science.mplstyle")
PHI_CHANNEL_INDEX = 0
FIELD_NAME = "Phi"
COLOR_MAP = "Blues"
QUIVER_STRIDE = 8
QUIVER_COLOR = "#08306b"

# User-editable run settings
MODEL_DIR: Path | None = Path("models") / "shortstep_0p1d_v1"
CHECKPOINT_PATH: Path | None = None
PROCESSED_DIR: Path | None = None
TEST_SHARD_INDEX = 0
INPUT_DAY = 0.0
TARGET_DAY = 0.1
ROLLOUT_STEP_DAYS: float | None = 0.1
DEVICE_MODE = "auto"
FIGURE_PATH: Path | None = None


def _dict_to_namespace(obj: Any) -> Any:
    """Convert nested dict/list structures into namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: _dict_to_namespace(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(value) for value in obj]
    return obj


def _resolve_repo_path(path: Path) -> Path:
    """Resolve repository-relative paths from the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _display_repo_path(path: Path) -> str:
    """Return a repository-relative path string for messages and metadata."""
    return str(Path(os.path.relpath(_resolve_repo_path(path), start=PROJECT_ROOT)))


def _resolve_checkpoint_path(*, model_dir: Path | None, checkpoint: Path | None) -> Path:
    """Resolve checkpoint path from top-level run settings."""
    if checkpoint is not None:
        resolved = _resolve_repo_path(checkpoint)
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {_display_repo_path(checkpoint)}")
        return resolved
    if model_dir is None:
        raise ValueError("Set MODEL_DIR or CHECKPOINT_PATH at the top of this file")
    ckpt_rel_path = Path(model_dir) / "best.pt"
    ckpt_path = _resolve_repo_path(ckpt_rel_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {_display_repo_path(ckpt_rel_path)}")
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
        processed_dir_path = Path(PROCESSED_DIR)
    else:
        resolved_config = ckpt.get("resolved_config")
        if not isinstance(resolved_config, dict):
            raise ValueError(
                "Checkpoint is missing resolved config metadata; set PROCESSED_DIR explicitly"
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
        processed_dir_path = Path(processed_dir_value)
    resolved = _resolve_repo_path(processed_dir_path)
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Processed directory not found: {_display_repo_path(processed_dir_path)}"
        )
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
        raise FileNotFoundError(f"Plot style not found: {_display_repo_path(STYLE_PATH)}")
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
    input_day: float,
    target_day: float,
    transition_days: float,
    rollout_step_days: float | None,
    n_rollout_steps: int,
    out_path: Path,
) -> None:
    """Save a 1x2 Phi comparison figure for one prediction."""
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
    rollout_suffix = ""
    if rollout_step_days is not None:
        rollout_suffix = (
            f" | rollout_step_days={float(rollout_step_days):.6f}"
            f" | rollout_steps={int(n_rollout_steps)}"
        )
    fig.suptitle(
        f"{shard_name} | input_day={input_day:.6f} | target_day={target_day:.6f} | "
        f"transition_days={transition_days:.6f}{rollout_suffix} | "
        f"{FIELD_NAME} RMSE={rmse:.3e}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _load_test_split_entries(processed_dir: Path) -> list[dict[str, Any]]:
    """Return held-out processed shard entries from the test split."""
    meta_path = (processed_dir / "processed_meta.json").resolve()
    if not meta_path.is_file():
        raise FileNotFoundError(f"Processed metadata not found: {_display_repo_path(meta_path)}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    split_entries = list(meta["splits"]["test"])
    if not split_entries:
        raise RuntimeError("Test split is empty")
    return split_entries


def _resolve_checkpoint_indices(
    *,
    input_time_days: float,
    target_time_days: float,
    checkpoint_days: np.ndarray,
) -> tuple[int, int, float, float]:
    """Map requested day-valued times onto saved checkpoint indices."""
    checkpoint_days = np.asarray(checkpoint_days, dtype=np.float64)
    if input_time_days < 0.0:
        raise ValueError("INPUT_DAY must be >= 0")
    if target_time_days <= input_time_days:
        raise ValueError("TARGET_DAY must be > INPUT_DAY")
    if target_time_days > float(checkpoint_days[-1]):
        raise ValueError(
            "TARGET_DAY exceeds the saved checkpoint horizon: "
            f"{target_time_days} > {float(checkpoint_days[-1])}"
        )

    interval_days = float(np.min(np.diff(checkpoint_days))) if checkpoint_days.size > 1 else 0.0
    tolerance_days = max(1.0e-12, 0.5 * interval_days)
    input_index = int(np.argmin(np.abs(checkpoint_days - float(input_time_days))))
    target_index = int(np.argmin(np.abs(checkpoint_days - float(target_time_days))))
    actual_input_day = float(checkpoint_days[input_index])
    actual_target_day = float(checkpoint_days[target_index])
    if abs(actual_input_day - float(input_time_days)) > tolerance_days:
        raise ValueError(
            "INPUT_DAY must align with a saved checkpoint within half the checkpoint cadence: "
            f"requested={float(input_time_days):.6f}, realized={actual_input_day:.6f}"
        )
    if abs(actual_target_day - float(target_time_days)) > tolerance_days:
        raise ValueError(
            "TARGET_DAY must align with a saved checkpoint within half the checkpoint cadence: "
            f"requested={float(target_time_days):.6f}, realized={actual_target_day:.6f}"
        )
    if target_index <= input_index:
        raise ValueError("Resolved target checkpoint must be later than the input checkpoint")
    return input_index, target_index, actual_input_day, actual_target_day


def _load_test_direct_jump_case(
    *,
    processed_dir: Path,
    stats: Any,
    test_shard_index: int,
    input_time_days: float,
    target_time_days: float,
) -> tuple[Any, np.ndarray, np.ndarray, str, float, float, float]:
    """Load one held-out direct-jump case.

    Returns:
        params:
            Physical conditioning parameters as ``Extended9Params``.
        initial_state_phys:
            Physical prognostic state with shape ``[3, H, W]``.
        true_target_phys:
            Physical prognostic target with shape ``[3, H, W]``.
        shard_name:
            Test-shard filename from ``processed_meta.json``.
        actual_input_day:
            Physical day represented by the resolved input step.
        actual_target_day:
            Physical day represented by the resolved target step.
        transition_days:
            Direct-jump horizon in physical days.
    """
    test_entries = _load_test_split_entries(processed_dir)
    if test_shard_index < 0 or test_shard_index >= len(test_entries):
        raise IndexError(
            f"TEST_SHARD_INDEX={test_shard_index} is out of range for {len(test_entries)} test shards"
        )

    shard_entry = dict(test_entries[test_shard_index])
    shard_name = str(shard_entry["file"])
    shard_path = (processed_dir / shard_name).resolve()
    if not shard_path.is_file():
        raise FileNotFoundError(f"Processed shard not found: {_display_repo_path(shard_path)}")

    with np.load(shard_path, allow_pickle=False) as npz:
        states_norm = np.asarray(npz["states_norm"], dtype=np.float32)
        params_norm = np.asarray(npz["params_norm"], dtype=np.float32)
        checkpoint_days = np.asarray(npz["checkpoint_days"], dtype=np.float64)

    params = to_extended9(_denormalize_params(params_norm, stats.params))
    input_index, target_index, actual_input_day, actual_target_day = _resolve_checkpoint_indices(
        input_time_days=float(input_time_days),
        target_time_days=float(target_time_days),
        checkpoint_days=checkpoint_days,
    )
    transition_days = float(actual_target_day - actual_input_day)
    initial_state_phys = denormalize_state_tensor(
        states_norm[input_index][None, ...],
        stats.state,
    )[0]
    true_target_phys = denormalize_state_tensor(
        states_norm[target_index][None, ...],
        stats.state,
    )[0]
    return (
        params,
        np.asarray(initial_state_phys, dtype=np.float32),
        np.asarray(true_target_phys, dtype=np.float32),
        shard_name,
        actual_input_day,
        actual_target_day,
        transition_days,
    )


def _predict_direct_jump(
    *,
    model: torch.nn.Module,
    stats: Any,
    params: Any,
    initial_state_phys: np.ndarray,
    transition_days: float,
    device: torch.device,
) -> np.ndarray:
    """Run one direct-jump model call and return a physical prognostic target.

    Args:
        initial_state_phys:
            Physical prognostic state with shape ``[3, H, W]``.
        transition_days:
            Requested direct-jump horizon in physical days.

    Returns:
        Physical prognostic prediction with shape ``[3, H, W]``.
    """
    params_vector = params_to_conditioning_vector(params)
    input_state_norm = normalize_state_tensor(
        initial_state_phys[None, ...],
        stats.state,
    ).astype(np.float32)
    conditioning_norm = normalize_conditioning(
        params_vector,
        np.asarray([transition_days], dtype=np.float64),
        param_stats=stats.params,
        transition_time_stats=stats.transition_time,
    ).astype(np.float32)

    with torch.inference_mode():
        pred_norm = model(
            torch.from_numpy(input_state_norm).to(device=device),
            torch.from_numpy(conditioning_norm).to(device=device),
        )
    return np.asarray(
        denormalize_state_tensor(
            pred_norm.detach().cpu().numpy(),
            stats.state,
        )[0],
        dtype=np.float32,
    )


def _build_rollout_schedule(
    *,
    transition_days: float,
    rollout_step_days: float | None,
) -> list[float]:
    """Return the sequence of jump durations used for prediction."""
    total_days = float(transition_days)
    if total_days <= 0.0:
        raise ValueError("transition_days must be > 0")
    if rollout_step_days is None:
        return [total_days]

    step_days = float(rollout_step_days)
    if step_days <= 0.0:
        raise ValueError("ROLLOUT_STEP_DAYS must be > 0 when set")
    if step_days > total_days:
        raise ValueError(
            "ROLLOUT_STEP_DAYS cannot exceed the requested transition horizon: "
            f"{step_days:.6f} > {total_days:.6f}"
        )

    n_rollout_steps = int(round(total_days / step_days))
    realized_days = float(n_rollout_steps) * step_days
    tolerance_days = max(1.0e-12, 1.0e-9 * max(abs(total_days), abs(step_days)))
    if n_rollout_steps < 1 or abs(realized_days - total_days) > tolerance_days:
        raise ValueError(
            "The requested target horizon must be an integer multiple of "
            "ROLLOUT_STEP_DAYS for fixed-step autoregressive rollout: "
            f"transition_days={total_days:.6f}, "
            f"rollout_step_days={step_days:.6f}"
        )
    return [step_days] * n_rollout_steps


def _predict_autoregressive_rollout(
    *,
    model: torch.nn.Module,
    stats: Any,
    params: Any,
    initial_state_phys: np.ndarray,
    transition_days: float,
    rollout_step_days: float | None,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    """Run repeated fixed-step predictions and return the final physical state."""
    rollout_schedule = _build_rollout_schedule(
        transition_days=float(transition_days),
        rollout_step_days=rollout_step_days,
    )
    current_state_phys = np.asarray(initial_state_phys, dtype=np.float32)
    for step_days in rollout_schedule:
        current_state_phys = _predict_direct_jump(
            model=model,
            stats=stats,
            params=params,
            initial_state_phys=current_state_phys,
            transition_days=float(step_days),
            device=device,
        )
    return np.asarray(current_state_phys, dtype=np.float32), int(len(rollout_schedule))


def _extract_target_winds(
    target_state: np.ndarray,
    *,
    target_field_names: Sequence[str],
    params: Any,
    solver_cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical-space winds from explicit channels or eta/delta diagnosis."""
    field_index = {str(name): idx for idx, name in enumerate(target_field_names)}
    if "U" in field_index and "V" in field_index:
        return (
            np.asarray(target_state[field_index["U"]], dtype=np.float64),
            np.asarray(target_state[field_index["V"]], dtype=np.float64),
        )
    return diagnose_winds(
        np.asarray(target_state[field_index["eta"]], dtype=np.float64),
        np.asarray(target_state[field_index["delta"]], dtype=np.float64),
        params=params,
        M=int(solver_cfg["M"]),
        dt_seconds=float(solver_cfg["dt_seconds"]),
    )


def main() -> None:
    """Load one held-out case, run the model, and save a plot."""
    _apply_plot_style()
    ckpt_path = _resolve_checkpoint_path(model_dir=MODEL_DIR, checkpoint=CHECKPOINT_PATH)
    run_dir = ckpt_path.parent
    figure_path = (
        _resolve_repo_path(FIGURE_PATH)
        if FIGURE_PATH is not None
        else (run_dir / "plots" / DEFAULT_FIGURE_NAME).resolve()
    )

    device = _resolve_device(DEVICE_MODE)
    ensure_torch_harmonics_importable()
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
        transition_days,
    ) = _load_test_direct_jump_case(
        processed_dir=processed_dir,
        stats=stats,
        test_shard_index=int(TEST_SHARD_INDEX),
        input_time_days=float(INPUT_DAY),
        target_time_days=float(TARGET_DAY),
    )

    model_cfg = _dict_to_namespace(ckpt["model_config"])
    shape = dict(ckpt["shape"])
    model = build_state_conditioned_transition_model(
        img_size=(int(shape["H"]), int(shape["W"])),
        input_state_chans=int(shape["C"]),
        target_state_chans=int(shape["C"]),
        param_dim=int(len(ckpt["conditioning_names"])),
        cfg_model=model_cfg,
        lat_order=str(ckpt["geometry"]["lat_order"]),
        lon_origin=str(ckpt["geometry"]["lon_origin"]),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device=device).eval()

    pred_target_phys, n_rollout_steps = _predict_autoregressive_rollout(
        model=model,
        stats=stats,
        params=params,
        initial_state_phys=initial_state_phys,
        transition_days=transition_days,
        rollout_step_days=ROLLOUT_STEP_DAYS,
        device=device,
    )
    solver_cfg = dict(ckpt["solver"])
    true_winds = _extract_target_winds(
        true_target_phys,
        target_field_names=stats.state.field_names,
        params=params,
        solver_cfg=solver_cfg,
    )
    pred_winds = _extract_target_winds(
        pred_target_phys,
        target_field_names=stats.state.field_names,
        params=params,
        solver_cfg=solver_cfg,
    )
    _save_figure(
        true_state=true_target_phys,
        pred_state=pred_target_phys,
        true_winds=true_winds,
        pred_winds=pred_winds,
        shard_name=shard_name,
        input_day=actual_input_day,
        target_day=actual_target_day,
        transition_days=transition_days,
        rollout_step_days=ROLLOUT_STEP_DAYS,
        n_rollout_steps=n_rollout_steps,
        out_path=figure_path,
    )
    if ROLLOUT_STEP_DAYS is None:
        print(f"Saved direct-jump prediction figure: {_display_repo_path(figure_path)}")
    else:
        print(f"Saved autoregressive rollout prediction figure: {_display_repo_path(figure_path)}")


if __name__ == "__main__":
    main()
