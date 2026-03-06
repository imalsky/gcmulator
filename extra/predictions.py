"""Visualize one predicted direct-jump prognostic transition from a trained GCMulator checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_MPL_CACHE_DIR = PROJECT_CACHE_DIR / "mplconfig"
PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_DIR.resolve()))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1")
MPL_CACHE_DIR = Path(
    os.environ.get("GCMULATOR_MPLCONFIGDIR", str(DEFAULT_MPL_CACHE_DIR))
).resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch


SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling import (
    build_state_conditioned_transition_model,
    ensure_torch_harmonics_importable,
)
from my_swamp_backend import (
    diagnose_winds,
    ensure_my_swamp_importable,
    params_to_conditioning_vector,
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
DEFAULT_FIGURE_NAME = "prognostic_true_vs_pred.png"
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
PROCESSED_DIR: Path | None = (PROJECT_ROOT / "data" / "processed").resolve()
SPLIT = "test"
SHARD_INDEX = 0
SAMPLE_INDEX = 0
COMPARE_DAY: float | None = None
DEVICE_MODE = "auto"
FIGURE_PATH: Path | None = None


def _dict_to_namespace(obj: Any) -> Any:
    """Convert nested dict/list structures into namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(
            **{key: _dict_to_namespace(value) for key, value in obj.items()}
        )
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
            raise RuntimeError("Requested --device=gpu but CUDA is unavailable")
        return torch.device("cuda")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device mode: {mode}")


def _resolve_processed_dir() -> Path:
    """Resolve processed directory from top-level settings."""
    if PROCESSED_DIR is not None:
        resolved = PROCESSED_DIR.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Processed directory not found: {resolved}")
        return resolved
    raise ValueError("Set PROCESSED_DIR at the top of this file")


def _target_step_from_day(compare_day: float, *, dt_seconds: float) -> int:
    """Convert an absolute day in the trajectory to an integer target step."""
    target_step = int(round(float(compare_day) * 86400.0 / float(dt_seconds)))
    if target_step < 1:
        raise ValueError("COMPARE_DAY must be at least one model step into the trajectory")
    return target_step


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


def _diagnose_target_winds(
    target_state: np.ndarray,
    *,
    target_field_names: tuple[str, ...],
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
        f"{shard_name} | {context_label} | "
        f"day={target_day:.6f} | "
        f"{FIELD_NAME} RMSE={rmse:.3e}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _load_comparison_example(
    *,
    ckpt: Dict[str, Any],
    processed_dir: Path,
    stats: Any,
    params: Any,
    shard_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """Build one input/target comparison example from either a shard sample or absolute day."""
    solver_cfg = dict(ckpt["solver"])

    if COMPARE_DAY is not None:
        target_step = _target_step_from_day(
            float(COMPARE_DAY),
            dt_seconds=float(solver_cfg["dt_seconds"]),
        )
        anchor_step = 0
        state_inputs_phys, state_targets_phys, _, _ = run_trajectory_window(
            params,
            M=int(solver_cfg["M"]),
            dt_seconds=float(solver_cfg["dt_seconds"]),
            time_days=float(COMPARE_DAY),
            starttime_index=int(solver_cfg["starttime_index"]),
            window_start_step=anchor_step,
            n_transitions=1,
            transition_jump_steps=int(target_step),
        )
        state0_norm = normalize_state_tensor(
            np.asarray(state_inputs_phys, dtype=np.float32),
            stats.input_state,
        ).astype(np.float32)
        conditioning_norm = normalize_conditioning(
            params_to_conditioning_vector(params),
            np.asarray(
                [float(target_step) * float(solver_cfg["dt_seconds"]) / 86400.0],
                dtype=np.float64,
            ),
            param_stats=stats.params,
            transition_time_stats=stats.transition_time,
        ).astype(np.float32)
        true_state = np.asarray(state_targets_phys[0], dtype=np.float32)
        target_day = float(target_step) * float(solver_cfg["dt_seconds"]) / 86400.0
        return (
            conditioning_norm,
            state0_norm,
            true_state,
            f"compare_day={target_day:.6f}",
            target_day,
        )

    shard_path = (processed_dir / shard_name).resolve()
    if not shard_path.is_file():
        raise FileNotFoundError(f"Processed shard not found: {shard_path}")

    with np.load(shard_path, allow_pickle=False) as npz:
        conditioning_norm = np.asarray(npz["conditioning_norm"], dtype=np.float32)
        state_inputs_norm = np.asarray(npz["state_inputs_norm"], dtype=np.float32)
        state_targets_norm = np.asarray(npz["state_targets_norm"], dtype=np.float32)
        transition_days = np.asarray(npz["transition_days"], dtype=np.float64)
        anchor_steps = np.asarray(npz["anchor_steps"], dtype=np.int64)
    if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= int(state_inputs_norm.shape[0]):
        raise IndexError(
            f"sample-index {SAMPLE_INDEX} is out of range "
            f"for shard '{shard_name}'"
        )

    state0_norm = state_inputs_norm[SAMPLE_INDEX : SAMPLE_INDEX + 1]
    true_state = denormalize_state_tensor(
        state_targets_norm[SAMPLE_INDEX : SAMPLE_INDEX + 1],
        stats.target_state,
    )[0]
    target_day = (
        float(anchor_steps[SAMPLE_INDEX]) * float(solver_cfg["dt_seconds"]) / 86400.0
        + float(transition_days[SAMPLE_INDEX])
    )
    return (
        conditioning_norm[SAMPLE_INDEX : SAMPLE_INDEX + 1],
        state0_norm,
        true_state,
        f"sample={int(SAMPLE_INDEX)}",
        target_day,
    )


def main() -> None:
    """Load one shard sample, run one direct-jump prediction, and save a prognostic plot."""
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

    processed_dir = _resolve_processed_dir()
    processed_meta_path = (processed_dir / "processed_meta.json").resolve()
    if not processed_meta_path.is_file():
        raise FileNotFoundError(f"Processed metadata not found: {processed_meta_path}")
    processed_meta = json.loads(processed_meta_path.read_text(encoding="utf-8"))

    split_entries = list(processed_meta["splits"][SPLIT])
    if not split_entries:
        raise RuntimeError(f"Split '{SPLIT}' is empty")
    if SHARD_INDEX < 0 or SHARD_INDEX >= len(split_entries):
        raise IndexError(
            f"shard-index {SHARD_INDEX} is out of range "
            f"for split '{SPLIT}'"
    )
    shard_entry = dict(split_entries[SHARD_INDEX])
    shard_name = str(shard_entry["file"])
    shard_path = (processed_dir / shard_name).resolve()
    if not shard_path.is_file():
        raise FileNotFoundError(f"Processed shard not found: {shard_path}")

    with np.load(shard_path, allow_pickle=False) as npz:
        params_norm = np.asarray(npz["params_norm"], dtype=np.float32)

    stats = stats_from_json(ckpt["normalization"])
    params_norm = params_norm[None, :]
    params = to_extended9(_denormalize_params(params_norm[0], stats.params))
    conditioning_norm, state0_norm, true_state, context_label, target_day = _load_comparison_example(
        ckpt=ckpt,
        processed_dir=processed_dir,
        stats=stats,
        params=params,
        shard_name=shard_name,
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

    with torch.inference_mode():
        pred_norm = model(
            torch.from_numpy(state0_norm).to(device=device),
            torch.from_numpy(conditioning_norm).to(device=device),
        )
    pred_phys = denormalize_state_tensor(pred_norm.detach().cpu().numpy(), stats.target_state)[0]
    true_winds = _diagnose_target_winds(
        true_state,
        target_field_names=stats.target_state.field_names,
        params=params,
        solver_cfg=dict(ckpt["solver"]),
    )
    pred_winds = _diagnose_target_winds(
        pred_phys,
        target_field_names=stats.target_state.field_names,
        params=params,
        solver_cfg=dict(ckpt["solver"]),
    )

    _save_figure(
        true_state=true_state,
        pred_state=pred_phys,
        true_winds=true_winds,
        pred_winds=pred_winds,
        shard_name=shard_name,
        context_label=context_label,
        target_day=float(target_day),
        out_path=figure_path,
    )
    print(f"Saved prognostic prediction figure: {figure_path}")


if __name__ == "__main__":
    main()
