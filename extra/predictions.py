"""Visualize one predicted one-step prognostic transition from a trained GCMulator checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict

MPL_CACHE_DIR = Path(os.environ.get("GCMULATOR_MPLCONFIGDIR", "/tmp/gcmulator_mplcache")).resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import resolve_path
from modeling import (
    build_state_conditioned_transition_model,
    ensure_torch_harmonics_importable,
)
from normalization import (
    denormalize_state_tensor,
    normalize_params,
    normalize_state_tensor,
    stats_from_json,
)


FIGURE_DPI = 180
DEFAULT_FIGURE_NAME = "prognostic_true_vs_pred.png"
STYLE_PATH = Path(__file__).resolve().with_name("science.mplstyle")
CHANNEL_NAMES = ("Phi", "eta", "delta")
COLOR_MAP = "coolwarm"


def _dict_to_namespace(obj: Any) -> Any:
    """Convert nested dict/list structures into namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(
            **{key: _dict_to_namespace(value) for key, value in obj.items()}
        )
    if isinstance(obj, list):
        return [_dict_to_namespace(value) for value in obj]
    return obj


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot one-step prognostic prediction from a trained "
            "GCMulator checkpoint"
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Run directory containing best.pt")
    source.add_argument("--checkpoint", type=Path, help="Checkpoint path to evaluate")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Processed split to read",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="Split shard index")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index within the shard")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Inference device",
    )
    parser.add_argument("--figure", type=Path, default=None, help="Explicit output figure path")
    return parser.parse_args()


def _resolve_checkpoint_path(*, run_dir: Path | None, checkpoint: Path | None) -> Path:
    """Resolve checkpoint path from CLI inputs."""
    if checkpoint is not None:
        resolved = checkpoint.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved
    if run_dir is None:
        raise ValueError("Either --run-dir or --checkpoint must be provided")
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


def _load_npy_payload_dict(file_path: Path) -> Dict[str, Any]:
    """Load a dict payload stored by ``np.save(..., allow_pickle=True)``."""
    obj = np.load(file_path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        payload = obj.item()
    else:
        payload = obj
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {file_path}, got {type(payload).__name__}")
    return payload


def _load_dataset_paths(ckpt: Dict[str, Any]) -> tuple[Path, Path]:
    """Resolve raw and processed dataset directories from checkpoint metadata."""
    resolved_config = dict(ckpt["resolved_config"])
    source_config_path = Path(str(ckpt["source_config_path"])).resolve()
    dataset_dir = resolve_path(source_config_path, str(resolved_config["paths"]["dataset_dir"]))
    processed_dir = resolve_path(source_config_path, str(resolved_config["paths"]["processed_dir"]))
    return dataset_dir, processed_dir


def _apply_plot_style() -> None:
    """Load the shared plotting style."""
    if not STYLE_PATH.is_file():
        raise FileNotFoundError(f"Plot style not found: {STYLE_PATH}")
    plt.style.use(str(STYLE_PATH))
    plt.rcParams["savefig.dpi"] = int(FIGURE_DPI)


def _channel_limit(true_field: np.ndarray, pred_field: np.ndarray) -> float:
    """Compute a symmetric robust color limit for one field."""
    values = np.concatenate([true_field.reshape(-1), pred_field.reshape(-1)]).astype(np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    limit = float(np.quantile(np.abs(values), 0.99))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.max(np.abs(values)))
    return max(limit, 1.0)


def _save_figure(
    *,
    true_state: np.ndarray,
    pred_state: np.ndarray,
    shard_name: str,
    sample_index: int,
    transition_days: float,
    out_path: Path,
) -> None:
    """Save a 2x3 prognostic comparison figure."""
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.0, 7.0),
        dpi=int(FIGURE_DPI),
        constrained_layout=True,
    )
    for channel_index, field_name in enumerate(CHANNEL_NAMES):
        true_field = np.asarray(true_state[channel_index], dtype=np.float64)
        pred_field = np.asarray(pred_state[channel_index], dtype=np.float64)
        vmax = _channel_limit(true_field, pred_field)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        im_true = axes[0, channel_index].imshow(
            true_field,
            origin="lower",
            cmap=COLOR_MAP,
            norm=norm,
            interpolation="bicubic",
            aspect="auto",
        )
        axes[1, channel_index].imshow(
            pred_field,
            origin="lower",
            cmap=COLOR_MAP,
            norm=norm,
            interpolation="bicubic",
            aspect="auto",
        )
        axes[0, channel_index].set_title(f"True {field_name}")
        axes[1, channel_index].set_title(f"Predicted {field_name}")
        axes[0, channel_index].set_xlabel("Longitude Index")
        axes[1, channel_index].set_xlabel("Longitude Index")
        axes[0, channel_index].set_ylabel("Latitude Index")
        axes[1, channel_index].set_ylabel("Latitude Index")
        fig.colorbar(im_true, ax=axes[:, channel_index], shrink=0.85)

    rmse = float(np.sqrt(np.mean((pred_state - true_state) ** 2)))
    fig.suptitle(
        f"{shard_name} | sample={sample_index} | "
        f"transition_days={transition_days:.6f} | "
        f"prognostic RMSE={rmse:.3e}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Load one shard sample, run one-step prediction, and save a prognostic plot."""
    _apply_plot_style()
    args = _parse_args()
    ckpt_path = _resolve_checkpoint_path(run_dir=args.run_dir, checkpoint=args.checkpoint)
    run_dir = ckpt_path.parent
    figure_path = (
        args.figure.resolve()
        if args.figure is not None
        else (run_dir / "plots" / DEFAULT_FIGURE_NAME).resolve()
    )

    device = _resolve_device(args.device)
    ensure_torch_harmonics_importable()
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=device)

    dataset_dir, processed_dir = _load_dataset_paths(ckpt)
    processed_meta_path = (processed_dir / "processed_meta.json").resolve()
    if not processed_meta_path.is_file():
        raise FileNotFoundError(f"Processed metadata not found: {processed_meta_path}")
    processed_meta = json.loads(processed_meta_path.read_text(encoding="utf-8"))

    split_entries = list(processed_meta["splits"][args.split])
    if not split_entries:
        raise RuntimeError(f"Split '{args.split}' is empty")
    if args.shard_index < 0 or args.shard_index >= len(split_entries):
        raise IndexError(
            f"shard-index {args.shard_index} is out of range "
            f"for split '{args.split}'"
        )
    shard_entry = dict(split_entries[args.shard_index])
    shard_name = str(shard_entry["file"])
    raw_file = (dataset_dir / f"{Path(shard_name).stem}.npy").resolve()
    if not raw_file.is_file():
        raise FileNotFoundError(f"Raw file for shard not found: {raw_file}")

    raw_payload = _load_npy_payload_dict(raw_file)
    state_inputs = np.asarray(raw_payload["state_inputs"], dtype=np.float32)
    state_targets = np.asarray(raw_payload["state_targets"], dtype=np.float32)
    transition_days = np.asarray(raw_payload["transition_days"], dtype=np.float64)
    raw_params = np.asarray(raw_payload["params"], dtype=np.float64)
    if args.sample_index < 0 or args.sample_index >= int(state_inputs.shape[0]):
        raise IndexError(
            f"sample-index {args.sample_index} is out of range "
            f"for shard '{shard_name}'"
        )

    stats = stats_from_json(ckpt["normalization"])
    state0_norm = normalize_state_tensor(
        state_inputs[args.sample_index : args.sample_index + 1],
        stats.input_state,
    ).astype(np.float32)
    params_norm = normalize_params(raw_params[None, :], stats.params).astype(np.float32)

    model_cfg = _dict_to_namespace(ckpt["model_config"])
    shape = dict(ckpt["shape"])
    input_fields = list(ckpt["input_fields"])
    target_fields = list(ckpt["target_fields"])
    residual_input_indices = [input_fields.index(field_name) for field_name in target_fields]
    model = build_state_conditioned_transition_model(
        img_size=(int(shape["H"]), int(shape["W"])),
        input_state_chans=int(shape["input_C"]),
        target_state_chans=int(shape["target_C"]),
        param_dim=int(len(ckpt["param_names"])),
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
            torch.from_numpy(params_norm).to(device=device),
        )
    pred_phys = denormalize_state_tensor(pred_norm.detach().cpu().numpy(), stats.target_state)[0]

    _save_figure(
        true_state=state_targets[args.sample_index],
        pred_state=pred_phys,
        shard_name=shard_name,
        sample_index=int(args.sample_index),
        transition_days=float(transition_days[args.sample_index]),
        out_path=figure_path,
    )
    print(f"Saved prognostic prediction figure: {figure_path}")


if __name__ == "__main__":
    main()
