"""Utility script to visualize Phi predictions from a trained checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

# Keep matplotlib cache in a writable temp path to avoid permission/cache issues.
MPL_CACHE_DIR = Path(os.environ.get("GCMULATOR_MPLCONFIGDIR", "/tmp/gcmulator_mplcache")).resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

try:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib import ticker as mticker
except Exception as exc:  # pragma: no cover
    raise RuntimeError("predictions.py requires matplotlib. Install it in your environment first.") from exc
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import time_days_to_rollout_steps
from src.modeling import build_rollout_model, ensure_torch_harmonics_importable
from src.normalization import denormalize_states, normalize_params, stats_from_json

# Global configuration.
MODEL_DIR = Path("models/model")
CHECKPOINT_NAME = "best.pt"

DEVICE_MODE = "auto"
SPLIT_NAME = "val"
PICKED_PROCESSED_NAME: str | None = None  # Example: "sim_000042_val.npz"
PLOTS_DIR_NAME = "plots"
FIGURE_NAME = "phi_true_vs_pred_max_days.png"
FIGURE_DPI = 180
PHI_COLOR_MAP = "coolwarm"
PHI_QUANTILE_CLIP = 0.01
PHI_SYMLOG_LIN_FRAC = 0.02
COLORBAR_WIDTH_RATIO = 0.07
STYLE_PATH = Path(__file__).resolve().with_name("science.mplstyle")


def _dict_to_namespace(obj: Any) -> Any:
    """Recursively convert nested dict/list structures into namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(x) for x in obj]
    return obj


def _resolve_model_dir(path_value: Path) -> Path:
    """Resolve model directory from absolute, repo-relative, or cwd-relative path."""
    if path_value.is_absolute():
        return path_value.resolve()

    # Default behavior: treat relative paths as repo-root relative.
    project_candidate = (PROJECT_ROOT / path_value).resolve()
    if project_candidate.exists():
        return project_candidate

    # Fallback: allow cwd-relative custom paths if the user set one.
    cwd_candidate = path_value.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return project_candidate


def _resolve_device(mode: str) -> torch.device:
    """Resolve plotting/inference device for this utility script."""
    m = str(mode).lower()
    if m == "cpu":
        return torch.device("cpu")
    if m == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("DEVICE_MODE='gpu' but CUDA is unavailable")
    if m == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"Unsupported DEVICE_MODE={mode}. Use 'cpu', 'gpu', or 'auto'.")


def _processed_to_raw_file_name(processed_name: str) -> str:
    """Map processed split filename back to corresponding raw simulation filename."""
    p = str(processed_name)
    if p.endswith("_train.npz"):
        return p[: -len("_train.npz")] + ".npz"
    if p.endswith("_val.npz"):
        return p[: -len("_val.npz")] + ".npz"
    raise ValueError(f"Unexpected processed filename format: {processed_name}")


def _apply_plot_style() -> None:
    """Load shared matplotlib style and configure figure DPI."""
    if not STYLE_PATH.is_file():
        raise FileNotFoundError(f"science.mplstyle not found: {STYLE_PATH}")
    plt.style.use(str(STYLE_PATH))
    plt.rcParams["savefig.dpi"] = int(FIGURE_DPI)


def _pick_max_days_sample(*, dataset_dir: Path, split_files: List[str]) -> Tuple[str, Path, float]:
    """Pick sample with maximal ``time_days`` from a processed split list."""
    if not split_files:
        raise ValueError("split_files is empty")

    best_processed = ""
    best_raw: Path | None = None
    best_days = float("-inf")

    for processed_name in sorted(split_files):
        raw_file = (dataset_dir / _processed_to_raw_file_name(processed_name)).resolve()
        if not raw_file.is_file():
            raise FileNotFoundError(f"Raw simulation file not found: {raw_file}")

        with np.load(raw_file, allow_pickle=True) as z:
            time_days = float(np.asarray(z["time_days"]).item())

        is_new_max = time_days > best_days + 1.0e-12
        is_tie_break = abs(time_days - best_days) <= 1.0e-12 and best_raw is not None and raw_file.name < best_raw.name
        if is_new_max or is_tie_break:
            best_processed = processed_name
            best_raw = raw_file
            best_days = time_days

    if best_raw is None or not np.isfinite(best_days):
        raise RuntimeError("Failed to select max-time_days sample")
    return best_processed, best_raw, best_days


def _robust_phi_signed_limit(true_phi: np.ndarray, pred_phi: np.ndarray) -> float:
    """Compute robust symmetric color limit for signed Phi plotting."""
    vals = np.concatenate([true_phi.reshape(-1), pred_phi.reshape(-1)]).astype(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite Phi values were found for plotting")
    abs_vals = np.abs(vals)
    vmax = float(np.quantile(abs_vals, 1.0 - PHI_QUANTILE_CLIP))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(abs_vals))
    if vmax <= 0.0:
        vmax = 1.0
    return vmax


def _save_phi_figure(
    *,
    true_phi: np.ndarray,
    pred_phi: np.ndarray,
    sim_name: str,
    time_days: float,
    steps: int,
    device: torch.device,
    out_path: Path,
) -> None:
    """Render side-by-side true/predicted Phi maps with shared signed-symlog colorbar."""
    vmax = _robust_phi_signed_limit(true_phi, pred_phi)
    linthresh = max(float(vmax) * float(PHI_SYMLOG_LIN_FRAC), np.finfo(np.float64).tiny)
    norm = mcolors.SymLogNorm(
        linthresh=linthresh,
        linscale=1.0,
        vmin=-float(vmax),
        vmax=float(vmax),
        base=10.0,
    )
    fig = plt.figure(
        figsize=(12.0, 4.8),
        dpi=int(FIGURE_DPI),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.0, 1.0, COLORBAR_WIDTH_RATIO],
        wspace=0.06,
    )
    ax_true = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    im_true = ax_true.imshow(
        true_phi,
        origin="lower",
        cmap=PHI_COLOR_MAP,
        norm=norm,
        interpolation="bicubic",
        aspect="auto",
    )
    ax_true.set_title("True Phi")
    ax_true.set_xlabel("Longitude Index")
    ax_true.set_ylabel("Latitude Index")

    im_pred = ax_pred.imshow(
        pred_phi,
        origin="lower",
        cmap=PHI_COLOR_MAP,
        norm=norm,
        interpolation="bicubic",
        aspect="auto",
    )
    ax_pred.set_title("Predicted Phi")
    ax_pred.set_xlabel("Longitude Index")
    ax_pred.set_ylabel("Latitude Index")

    cbar = fig.colorbar(im_pred, cax=cax)
    cbar.set_label("Phi (signed symlog)")
    cbar.locator = mticker.MaxNLocator(nbins=6)
    cbar.update_ticks()

    rmse_phi = float(np.sqrt(np.mean((pred_phi - true_phi) ** 2)))
    fig.suptitle(
        f"{sim_name} | time_days={time_days:.3f} | steps={steps} | device={device} | Phi RMSE={rmse_phi:.3e} | signed symlog"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Load checkpoint, run one-sample prediction, and save Phi comparison figure."""
    _apply_plot_style()

    model_dir = _resolve_model_dir(MODEL_DIR)
    ckpt_path = (model_dir / CHECKPOINT_NAME).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            f"Set MODEL_DIR to a valid run folder (for example under {(PROJECT_ROOT / 'models').resolve()})."
        )

    device = _resolve_device(DEVICE_MODE)
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=device)
    source_cfg_path = Path(str(ckpt["source_config_path"])).resolve()
    ensure_torch_harmonics_importable(source_cfg_path.parent)

    model_cfg = _dict_to_namespace(ckpt["model_config"])
    shape = dict(ckpt["shape"])
    state_chans = int(shape["C"])
    h = int(shape["H"])
    w = int(shape["W"])
    param_dim = int(len(ckpt["param_names"]))
    fields = list(ckpt["fields"])

    source_cfg_dir = source_cfg_path.parent
    resolved_cfg = dict(ckpt["resolved_config"])
    dataset_dir = (source_cfg_dir / str(resolved_cfg["paths"]["dataset_dir"])).resolve()
    processed_dir = (source_cfg_dir / str(resolved_cfg["paths"]["processed_dir"])).resolve()
    processed_meta_path = (processed_dir / "processed_meta.json").resolve()
    if not processed_meta_path.is_file():
        raise FileNotFoundError(f"processed_meta.json not found: {processed_meta_path}")

    processed_meta = json.loads(processed_meta_path.read_text(encoding="utf-8"))
    split_files = list(processed_meta["splits"][SPLIT_NAME])
    if not split_files:
        raise RuntimeError(f"No files in processed split '{SPLIT_NAME}' at {processed_meta_path}")

    if PICKED_PROCESSED_NAME is None:
        picked_processed, raw_file, time_days = _pick_max_days_sample(
            dataset_dir=dataset_dir,
            split_files=split_files,
        )
    else:
        if PICKED_PROCESSED_NAME not in split_files:
            raise ValueError(
                f"PICKED_PROCESSED_NAME={PICKED_PROCESSED_NAME!r} is not in split '{SPLIT_NAME}'. "
                f"Available count={len(split_files)}"
            )
        picked_processed = str(PICKED_PROCESSED_NAME)
        raw_file = (dataset_dir / _processed_to_raw_file_name(picked_processed)).resolve()
        with np.load(raw_file, allow_pickle=True) as z:
            time_days = float(np.asarray(z["time_days"]).item())

    with np.load(raw_file, allow_pickle=True) as z:
        true_state = np.asarray(z["state_final"], dtype=np.float32)  # [C,H,W]
        raw_params = np.asarray(z["params"], dtype=np.float64)  # [P]

    if true_state.shape != (state_chans, h, w):
        raise ValueError(f"Unexpected state shape in {raw_file}: {true_state.shape} vs expected {(state_chans, h, w)}")

    stats = stats_from_json(ckpt["normalization"])
    params_norm = normalize_params(raw_params[None, :], stats=stats).astype(np.float32)
    params_t = torch.from_numpy(params_norm).to(device=device)

    steps = time_days_to_rollout_steps(
        time_days=time_days,
        default_time_days=float(ckpt["solver"]["default_time_days"]),
        rollout_steps_at_default_time=int(ckpt["model_config"]["rollout_steps_at_default_time"]),
    )

    model = build_rollout_model(
        img_size=(h, w),
        state_chans=state_chans,
        param_dim=param_dim,
        cfg_model=model_cfg,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device=device)
    model.eval()

    with torch.inference_mode():
        pred_norm_t = model(params_t, steps=steps)
        pred_norm = pred_norm_t.detach().cpu().numpy().astype(np.float32)

    pred_state = denormalize_states(pred_norm, stats=stats)[0]

    if "Phi" not in fields:
        raise ValueError(f"'Phi' not found in checkpoint fields: {fields}")
    phi_idx = int(fields.index("Phi"))
    true_phi = np.asarray(true_state[phi_idx], dtype=np.float32)
    pred_phi = np.asarray(pred_state[phi_idx], dtype=np.float32)
    true_nonpos = int(np.count_nonzero(true_phi <= 0.0))
    pred_nonpos = int(np.count_nonzero(pred_phi <= 0.0))

    out_path = (model_dir / PLOTS_DIR_NAME / FIGURE_NAME).resolve()
    _save_phi_figure(
        true_phi=true_phi,
        pred_phi=pred_phi,
        sim_name=raw_file.name,
        time_days=time_days,
        steps=int(steps),
        device=device,
        out_path=out_path,
    )

    print(f"Saved figure: {out_path}")
    print(f"Picked split file with max time_days: {picked_processed}")
    print(f"Raw simulation: {raw_file} | time_days={time_days:.6f}")
    print(
        f"True Phi range: [{float(np.min(true_phi)):.6e}, {float(np.max(true_phi)):.6e}] | non-positive cells={true_nonpos}"
    )
    print(
        f"Pred Phi range: [{float(np.min(pred_phi)):.6e}, {float(np.max(pred_phi)):.6e}] | non-positive cells={pred_nonpos}"
    )


if __name__ == "__main__":
    main()
