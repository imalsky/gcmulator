"""Utility script to visualize Phi predictions from a trained checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
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

from src.modeling import build_state_conditioned_rollout_model, ensure_torch_harmonics_importable
from src.normalization import denormalize_states, normalize_params, normalize_states, stats_from_json

# Global configuration.
MODEL_DIR = Path("models/model")
CHECKPOINT_NAME = "best.pt"

DEVICE_MODE = "auto"
SPLIT_NAME = "test"
PICKED_PROCESSED_NAME: str | None = None  # Example: "test/sim_000042_tr0003.npy"
ROLLOUT_STEPS_OVERRIDE: int | None = None  # Optional transition rollout override.
PLOTS_DIR_NAME = "plots"
FIGURE_NAME = "phi_true_vs_pred_max_days.png"
FIGURE_DPI = 180
PHI_COLOR_MAP = "coolwarm"
PHI_QUANTILE_CLIP = 0.01
COLORBAR_WIDTH_RATIO = 0.07
WIND_QUIVER_STRIDE = 4
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


def _resolve_source_cfg_dir(source_cfg_value: Any) -> Path:
    """Resolve checkpoint source config directory with local-project fallback."""
    source_cfg_path = Path(str(source_cfg_value)).expanduser()
    if source_cfg_path.is_file():
        return source_cfg_path.resolve().parent

    source_cfg_path_abs = source_cfg_path.resolve()
    if source_cfg_path_abs.is_file():
        return source_cfg_path_abs.parent

    return PROJECT_ROOT


def _resolve_dataset_and_processed_dirs(
    *,
    resolved_cfg: Dict[str, Any],
    source_cfg_dir: Path,
    model_dir: Path,
) -> Tuple[Path, Path]:
    """Resolve dataset/processed dirs and fall back to local roots if needed."""
    paths_cfg = dict(resolved_cfg.get("paths", {}))
    if "dataset_dir" not in paths_cfg or "processed_dir" not in paths_cfg:
        raise KeyError("resolved_config.paths must include dataset_dir and processed_dir")

    dataset_token = str(paths_cfg["dataset_dir"])
    processed_token = str(paths_cfg["processed_dir"])
    base_candidates = [source_cfg_dir, PROJECT_ROOT, model_dir.parent.parent, Path.cwd()]

    checked_meta_paths: List[Path] = []
    seen_bases: set[str] = set()
    for base in base_candidates:
        base_resolved = base.resolve()
        base_key = str(base_resolved)
        if base_key in seen_bases:
            continue
        seen_bases.add(base_key)

        dataset_dir = (base_resolved / dataset_token).resolve()
        processed_dir = (base_resolved / processed_token).resolve()
        processed_meta_path = (processed_dir / "processed_meta.json").resolve()
        checked_meta_paths.append(processed_meta_path)
        if processed_meta_path.is_file():
            return dataset_dir, processed_dir

    checked = ", ".join(str(p) for p in checked_meta_paths)
    raise FileNotFoundError(f"processed_meta.json not found. Checked: {checked}")


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
    name = Path(str(processed_name)).name
    m = re.match(r"^(sim_\d+)_tr\d+\.(npy|npz)$", name)
    if m is not None:
        return f"{m.group(1)}.{m.group(2)}"
    if name.endswith("_train.npy"):
        return name[: -len("_train.npy")] + ".npy"
    if name.endswith("_val.npy"):
        return name[: -len("_val.npy")] + ".npy"
    if name.endswith("_test.npy"):
        return name[: -len("_test.npy")] + ".npy"
    if name.endswith("_train.npz"):
        return name[: -len("_train.npz")] + ".npz"
    if name.endswith("_val.npz"):
        return name[: -len("_val.npz")] + ".npz"
    if name.endswith("_test.npz"):
        return name[: -len("_test.npz")] + ".npz"
    if name.startswith("sim_") and name.endswith(".npy"):
        return name
    if name.startswith("sim_") and name.endswith(".npz"):
        return name
    raise ValueError(f"Unexpected processed filename format: {processed_name}")


def _load_sim_payload(path: Path) -> Dict[str, Any]:
    """Load raw simulation payload from .npy (preferred) or legacy .npz."""
    if path.suffix.lower() == ".npy":
        obj = np.load(path, allow_pickle=True)
        payload = obj.item() if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object else obj
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in {path}, got {type(payload).__name__}")
        return payload
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _processed_transition_index(processed_name: str) -> int:
    """Extract transition index from processed filename."""
    name = Path(str(processed_name)).name
    m = re.search(r"_tr(\d+)\.(npy|npz)$", name)
    if m is None:
        return 0
    return int(m.group(1))


def _build_conditioning_vector(
    *,
    raw_params: np.ndarray,
    transition_days: float,
    stats,
    checkpoint: Dict[str, Any],
) -> np.ndarray:
    """Build normalized conditioning vector expected by the trained model."""
    params_norm = normalize_params(raw_params[None, :], stats=stats).astype(np.float32)[0]
    conditioning_names = list(checkpoint.get("conditioning_names", checkpoint["param_names"]))
    if len(conditioning_names) == int(params_norm.shape[0]):
        return params_norm.astype(np.float32, copy=False)

    if (
        len(conditioning_names) == int(params_norm.shape[0]) + 1
        and conditioning_names[-1] == "transition_days"
    ):
        transition_norm = dict(checkpoint.get("transition_days_norm", {}))
        if not transition_norm:
            raise ValueError("Checkpoint expects transition_days conditioning but lacks transition_days_norm metadata.")
        td_mean = float(transition_norm["mean"])
        td_std = float(transition_norm["std"])
        td_eps = float(transition_norm["zscore_eps"])
        td_is_constant = bool(transition_norm["is_constant"])
        td_norm = 0.0 if td_is_constant else (float(transition_days) - td_mean) / (td_std + td_eps)
        return np.concatenate(
            [params_norm.astype(np.float32, copy=False), np.asarray([td_norm], dtype=np.float32)],
            axis=0,
        )

    raise ValueError(
        "Unsupported conditioning schema. "
        f"conditioning_names={conditioning_names}, param_names={list(checkpoint['param_names'])}."
    )


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

        z = _load_sim_payload(raw_file)
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
    """Compute robust symmetric color limit for centered Phi plotting."""
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
    true_u: np.ndarray,
    true_v: np.ndarray,
    pred_u: np.ndarray,
    pred_v: np.ndarray,
    sim_name: str,
    time_days: float,
    steps: int,
    device: torch.device,
    out_path: Path,
) -> None:
    """Render side-by-side true/predicted Phi maps with wind quiver overlays."""
    vmax = _robust_phi_signed_limit(true_phi, pred_phi)
    norm = mcolors.TwoSlopeNorm(
        vmin=-float(vmax),
        vcenter=0.0,
        vmax=float(vmax),
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
    stride = max(1, int(WIND_QUIVER_STRIDE))
    h, w = true_phi.shape
    y_idx = np.arange(0, h, stride, dtype=np.int32)
    x_idx = np.arange(0, w, stride, dtype=np.int32)
    xx, yy = np.meshgrid(x_idx, y_idx)
    ax_true.quiver(
        xx,
        yy,
        true_u[yy, xx],
        true_v[yy, xx],
        color="white",
        pivot="mid",
        angles="xy",
        scale_units="xy",
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
    ax_pred.quiver(
        xx,
        yy,
        pred_u[yy, xx],
        pred_v[yy, xx],
        color="white",
        pivot="mid",
        angles="xy",
        scale_units="xy",
    )
    ax_pred.set_title("Predicted Phi")
    ax_pred.set_xlabel("Longitude Index")
    ax_pred.set_ylabel("Latitude Index")

    cbar = fig.colorbar(im_pred, cax=cax)
    cbar.set_label("Phi (linear)")
    cbar.locator = mticker.MaxNLocator(nbins=7, symmetric=True)
    cbar.update_ticks()

    rmse_phi = float(np.sqrt(np.mean((pred_phi - true_phi) ** 2)))
    fig.suptitle(
        f"{sim_name} | time_days={time_days:.3f} | transitions={steps} | device={device} | Phi RMSE={rmse_phi:.3e}"
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
    source_cfg_dir = _resolve_source_cfg_dir(ckpt["source_config_path"])
    ensure_torch_harmonics_importable(source_cfg_dir)

    model_cfg = _dict_to_namespace(ckpt["model_config"])
    ckpt_geometry = dict(ckpt.get("geometry", {}))
    lat_order = str(ckpt_geometry.get("lat_order", "north_to_south"))
    lon_origin = str(ckpt_geometry.get("lon_origin", "0_to_2pi"))
    shape = dict(ckpt["shape"])
    state_chans = int(shape["C"])
    h = int(shape["H"])
    w = int(shape["W"])
    conditioning_names = list(ckpt.get("conditioning_names", ckpt["param_names"]))
    conditioning_dim = int(len(conditioning_names))
    fields = list(ckpt["fields"])

    resolved_cfg = dict(ckpt["resolved_config"])
    dataset_dir, processed_dir = _resolve_dataset_and_processed_dirs(
        resolved_cfg=resolved_cfg,
        source_cfg_dir=source_cfg_dir,
        model_dir=model_dir,
    )
    processed_meta_path = (processed_dir / "processed_meta.json").resolve()
    if not processed_meta_path.is_file():
        raise FileNotFoundError(f"processed_meta.json not found: {processed_meta_path}")

    processed_meta = json.loads(processed_meta_path.read_text(encoding="utf-8"))
    split_map = dict(processed_meta["splits"])
    if SPLIT_NAME not in split_map:
        raise KeyError(f"Split '{SPLIT_NAME}' not found in processed_meta.json. Available: {list(split_map.keys())}")
    split_files = list(split_map[SPLIT_NAME])
    if not split_files:
        raise RuntimeError(f"No files in processed split '{SPLIT_NAME}' at {processed_meta_path}")

    if PICKED_PROCESSED_NAME is None:
        picked_processed = sorted(split_files)[0]
    else:
        if PICKED_PROCESSED_NAME not in split_files:
            raise ValueError(
                f"PICKED_PROCESSED_NAME={PICKED_PROCESSED_NAME!r} is not in split '{SPLIT_NAME}'. "
                f"Available count={len(split_files)}"
            )
        picked_processed = str(PICKED_PROCESSED_NAME)
    raw_file = (dataset_dir / _processed_to_raw_file_name(picked_processed)).resolve()
    z = _load_sim_payload(raw_file)
    time_days = float(np.asarray(z["time_days"]).item())
    transition_index = _processed_transition_index(picked_processed)

    z = _load_sim_payload(raw_file)
    state_inputs = np.asarray(z["state_inputs"], dtype=np.float32)  # [T,C,H,W]
    state_targets = np.asarray(z["state_targets"], dtype=np.float32)  # [T,C,H,W]
    transition_days = np.asarray(z["transition_days"], dtype=np.float64)  # [T]
    raw_params = np.asarray(z["params"], dtype=np.float64)  # [P]
    if state_inputs.ndim != 4 or state_targets.ndim != 4:
        raise ValueError(f"Raw trajectory tensors must be [T,C,H,W], got {state_inputs.shape} and {state_targets.shape}")
    if state_inputs.shape != state_targets.shape:
        raise ValueError(f"Raw trajectory input/target mismatch: {state_inputs.shape} vs {state_targets.shape}")
    if transition_days.ndim != 1 or int(transition_days.shape[0]) != int(state_inputs.shape[0]):
        raise ValueError(f"transition_days shape mismatch: {transition_days.shape} for T={state_inputs.shape[0]}")
    if transition_index < 0 or transition_index >= int(state_inputs.shape[0]):
        raise ValueError(
            f"Transition index {transition_index} is out of bounds for trajectory length {int(state_inputs.shape[0])}"
        )

    if state_inputs.shape[1:] != (state_chans, h, w):
        raise ValueError(
            f"Unexpected trajectory state shape in {raw_file}: {state_inputs.shape[1:]} vs expected {(state_chans, h, w)}"
        )

    stats = stats_from_json(ckpt["normalization"])
    state0_norm = normalize_states(state_inputs[transition_index : transition_index + 1], stats=stats).astype(np.float32)
    transition_days_value = float(transition_days[transition_index])
    conditioning_norm = _build_conditioning_vector(
        raw_params=raw_params,
        transition_days=transition_days_value,
        stats=stats,
        checkpoint=ckpt,
    )
    conditioning_t = torch.from_numpy(conditioning_norm[None, ...]).to(device=device)
    state_t = torch.from_numpy(state0_norm).to(device=device)

    steps = 1 if ROLLOUT_STEPS_OVERRIDE is None else int(ROLLOUT_STEPS_OVERRIDE)
    if steps != 1:
        raise ValueError(
            "This utility currently supports one direct transition per call (steps=1). "
            f"Got ROLLOUT_STEPS_OVERRIDE={ROLLOUT_STEPS_OVERRIDE}."
        )

    model = build_state_conditioned_rollout_model(
        img_size=(h, w),
        state_chans=state_chans,
        param_dim=conditioning_dim,
        cfg_model=model_cfg,
        lat_order=lat_order,
        lon_origin=lon_origin,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device=device)
    model.eval()

    with torch.inference_mode():
        pred_norm_t = model(state_t, conditioning_t, steps=1)
        pred_norm = pred_norm_t.detach().cpu().numpy().astype(np.float32)

    true_state = np.asarray(state_targets[transition_index], dtype=np.float32)
    pred_state = denormalize_states(pred_norm, stats=stats)[0]
    total_days = float(transition_days_value)

    if "Phi" not in fields:
        raise ValueError(f"'Phi' not found in checkpoint fields: {fields}")
    if "U" not in fields or "V" not in fields:
        raise ValueError(f"'U' and 'V' must be present in checkpoint fields: {fields}")
    phi_idx = int(fields.index("Phi"))
    u_idx = int(fields.index("U"))
    v_idx = int(fields.index("V"))
    true_phi = np.asarray(true_state[phi_idx], dtype=np.float32)
    true_u = np.asarray(true_state[u_idx], dtype=np.float32)
    true_v = np.asarray(true_state[v_idx], dtype=np.float32)
    pred_phi = np.asarray(pred_state[phi_idx], dtype=np.float32)
    pred_u = np.asarray(pred_state[u_idx], dtype=np.float32)
    pred_v = np.asarray(pred_state[v_idx], dtype=np.float32)
    true_nonpos = int(np.count_nonzero(true_phi <= 0.0))
    pred_nonpos = int(np.count_nonzero(pred_phi <= 0.0))

    out_path = (model_dir / PLOTS_DIR_NAME / FIGURE_NAME).resolve()
    _save_phi_figure(
        true_phi=true_phi,
        pred_phi=pred_phi,
        true_u=true_u,
        true_v=true_v,
        pred_u=pred_u,
        pred_v=pred_v,
        sim_name=raw_file.name,
        time_days=total_days,
        steps=int(steps),
        device=device,
        out_path=out_path,
    )

    print(f"Saved figure: {out_path}")
    print(f"Picked split sample: {picked_processed} (transition index={transition_index})")
    print(f"Raw simulation: {raw_file} | time_days={time_days:.6f} | evaluated_days={total_days:.6f}")
    print(
        f"True Phi range: [{float(np.min(true_phi)):.6e}, {float(np.max(true_phi)):.6e}] | non-positive cells={true_nonpos}"
    )
    print(
        f"Pred Phi range: [{float(np.min(pred_phi)):.6e}, {float(np.max(pred_phi)):.6e}] | non-positive cells={pred_nonpos}"
    )


if __name__ == "__main__":
    main()
