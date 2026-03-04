"""Parity utility comparing terminal SWAMPE and MY_SWAMP trajectories."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Dict

import numpy as np
import scipy.special as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep matplotlib cache in a writable temp path to avoid permission/cache issues.
MPL_CACHE_DIR = Path(os.environ.get("GCMULATOR_MPLCONFIGDIR", "/tmp/gcmulator_mplcache")).resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

# Force CPU-only execution and preserve float64 parity behavior.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1")

try:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib import ticker as mticker
except Exception as exc:  # pragma: no cover
    raise RuntimeError("swampe_parity_compare.py requires matplotlib. Install it in your environment first.") from exc

from src.config import Extended9Params
from src.my_swamp_backend import ensure_my_swamp_importable

# ---------------------------------------------------------------------------
# User-editable run settings
# ---------------------------------------------------------------------------
M = 42
DT_SECONDS = 1200.0
TIME_DAYS = 5.0
STARTTIME_INDEX = 2

PARAMS = Extended9Params(
    a_m=8.2e7,
    omega_rad_s=3.2e-5,
    Phibar=3.0e5,
    DPhieq=1.0e6,
    taurad_s=10.0 * 3600.0,
    taudrag_s=6.0 * 3600.0,
    g_m_s2=9.8,
    K6=1.24e33,
    K6Phi=None,
)

ALPHA = 0.01
DIFFFLAG = True
FORCFLAG = True
MODALFLAG = True
EXPFLAG = False
TEST_CASE = None
JIT_SCAN = True

OUT_DIR = (PROJECT_ROOT / "extra" / "parity_outputs").resolve()
_time_tag = f"{TIME_DAYS:.3f}".rstrip("0").rstrip(".").replace(".", "p")
FIGURE_NAME = f"phi_swampe_vs_my_swamp_{_time_tag}d.png"
REPORT_NAME = f"swampe_vs_my_swamp_{_time_tag}d_metrics.json"

ATOL_BY_FIELD: Dict[str, float] = {
    "Phi": 5.0e-8,
    "U": 1.0e-9,
    "V": 1.0e-9,
    "eta": 1.0e-10,
    "delta": 1.0e-10,
}
FAIL_ON_TOLERANCE = True

# ---------------------------------------------------------------------------
# Plot style settings
# ---------------------------------------------------------------------------
STYLE_PATH = Path(__file__).resolve().with_name("science.mplstyle")
FIGURE_DPI = 180
PHI_COLOR_MAP = "coolwarm"
PHI_QUANTILE_CLIP = 0.01
PHI_SYMLOG_LIN_FRAC = 0.02
COLORBAR_WIDTH_RATIO = 0.07


def _ensure_lpmn_compat() -> None:
    """Provide SciPy ``lpmn`` compatibility shim when missing."""
    if hasattr(sp, "lpmn"):
        return

    def _lpmn_compat(m_max: int, n_max: int, x: float):
        m_max = int(m_max)
        n_max = int(n_max)
        p = np.zeros((m_max + 1, n_max + 1), dtype=np.float64)
        dp = np.zeros((m_max + 1, n_max + 1), dtype=np.float64)
        p[0, 0] = 1.0

        s = math.sqrt(max(0.0, 1.0 - x * x))
        for m_i in range(1, m_max + 1):
            if m_i <= n_max:
                p[m_i, m_i] = -(2 * m_i - 1) * s * p[m_i - 1, m_i - 1]

        for m_i in range(0, min(m_max, n_max - 1) + 1):
            p[m_i, m_i + 1] = (2 * m_i + 1) * x * p[m_i, m_i]

        for m_i in range(0, m_max + 1):
            for n_i in range(m_i + 2, n_max + 1):
                p[m_i, n_i] = ((2 * n_i - 1) * x * p[m_i, n_i - 1] - (n_i + m_i - 1) * p[m_i, n_i - 2]) / (
                    n_i - m_i
                )

        denom = x * x - 1.0
        if denom == 0.0:
            denom = np.finfo(np.float64).eps

        for m_i in range(0, m_max + 1):
            if m_i <= n_max:
                dp[m_i, m_i] = 0.0 if m_i == 0 else (m_i * x * p[m_i, m_i]) / denom
            for n_i in range(m_i + 1, n_max + 1):
                dp[m_i, n_i] = (n_i * x * p[m_i, n_i] - (n_i + m_i) * p[m_i, n_i - 1]) / denom
        return p, dp

    sp.lpmn = _lpmn_compat


def _ensure_swampe_importable() -> None:
    """Ensure SWAMPE import succeeds from install or adjacent checkout."""
    try:
        import SWAMPE.continuation  # noqa: F401
        import SWAMPE.model  # noqa: F401
        return
    except Exception:
        pass

    candidates = [
        PROJECT_ROOT / "SWAMPE",
        PROJECT_ROOT.parent / "SWAMPE",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
            try:
                import SWAMPE.continuation  # noqa: F401
                import SWAMPE.model  # noqa: F401
                return
            except Exception:
                sys.path.pop(0)

    raise RuntimeError("Could not import SWAMPE. Install it or place SWAMPE/ next to this repository.")


def _apply_plot_style() -> None:
    """Load shared plotting style for parity figures."""
    if not STYLE_PATH.is_file():
        raise FileNotFoundError(f"science.mplstyle not found: {STYLE_PATH}")
    plt.style.use(str(STYLE_PATH))
    plt.rcParams["savefig.dpi"] = int(FIGURE_DPI)


def _robust_phi_signed_limit(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    """Compute robust symmetric plotting range for signed Phi fields."""
    vals = np.concatenate([phi_a.reshape(-1), phi_b.reshape(-1)]).astype(np.float64)
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
    swampe_phi: np.ndarray,
    my_swamp_phi: np.ndarray,
    out_path: Path,
    steps: int,
    compared_time_days: float,
) -> None:
    """Save side-by-side SWAMPE/MY_SWAMP Phi comparison figure."""
    vmax = _robust_phi_signed_limit(swampe_phi, my_swamp_phi)
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
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_new = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    ax_ref.imshow(
        swampe_phi,
        origin="lower",
        cmap=PHI_COLOR_MAP,
        norm=norm,
        interpolation="bicubic",
        aspect="auto",
    )
    ax_ref.set_title("SWAMPE Phi")
    ax_ref.set_xlabel("Longitude Index")
    ax_ref.set_ylabel("Latitude Index")

    im_new = ax_new.imshow(
        my_swamp_phi,
        origin="lower",
        cmap=PHI_COLOR_MAP,
        norm=norm,
        interpolation="bicubic",
        aspect="auto",
    )
    ax_new.set_title("MY_SWAMP Phi")
    ax_new.set_xlabel("Longitude Index")
    ax_new.set_ylabel("Latitude Index")

    cbar = fig.colorbar(im_new, cax=cax)
    cbar.set_label("Phi (signed symlog)")
    cbar.locator = mticker.MaxNLocator(nbins=6)
    cbar.update_ticks()

    diff = np.asarray(my_swamp_phi, dtype=np.float64) - np.asarray(swampe_phi, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    fig.suptitle(
        f"SWAMPE vs MY_SWAMP | time_days={compared_time_days:.3f} | steps={steps} | CPU | "
        f"Phi RMSE={rmse:.3e} | max_abs={max_abs:.3e}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _last_saved_timestamp_seconds(path: Path) -> int:
    """Find most recent SWAMPE snapshot timestamp in save directory."""
    stamps: list[int] = []
    for p in path.glob("Phi-*"):
        try:
            stamps.append(int(p.name.split("Phi-", 1)[1]))
        except Exception:
            continue
    if not stamps:
        raise RuntimeError(f"SWAMPE did not save any Phi-* snapshots in {path}")
    return int(max(stamps))


def _run_swampe(*, tmax: int) -> Dict[str, object]:
    """Run SWAMPE and return terminal fields plus progression metadata."""
    _ensure_lpmn_compat()
    _ensure_swampe_importable()
    import SWAMPE.continuation as swampe_cont
    import SWAMPE.model as swampe_model

    with tempfile.TemporaryDirectory(prefix="swampe_parity_", dir="/tmp") as tmp:
        custompath = f"{tmp}/"
        swampe_model.run_model(
            M=int(M),
            dt=float(DT_SECONDS),
            tmax=int(tmax),
            Phibar=float(PARAMS.Phibar),
            omega=float(PARAMS.omega_rad_s),
            a=float(PARAMS.a_m),
            test=TEST_CASE,
            g=float(PARAMS.g_m_s2),
            forcflag=FORCFLAG,
            taurad=float(PARAMS.taurad_s),
            taudrag=float(PARAMS.taudrag_s),
            DPhieq=float(PARAMS.DPhieq),
            plotflag=False,
            diffflag=DIFFFLAG,
            modalflag=MODALFLAG,
            alpha=float(ALPHA),
            saveflag=True,
            expflag=EXPFLAG,
            savefreq=1,
            K6=float(PARAMS.K6),
            custompath=custompath,
            timeunits="seconds",
            verbose=False,
        )

        last_ts_s = _last_saved_timestamp_seconds(Path(custompath))
        terminal_step = int(round(float(last_ts_s) / float(DT_SECONDS)))
        target_step = int(tmax - 1)
        timestamp = str(int(last_ts_s))
        return {
            "Phi": np.asarray(swampe_cont.read_pickle(f"Phi-{timestamp}", custompath=custompath), dtype=np.float64),
            "U": np.asarray(swampe_cont.read_pickle(f"U-{timestamp}", custompath=custompath), dtype=np.float64),
            "V": np.asarray(swampe_cont.read_pickle(f"V-{timestamp}", custompath=custompath), dtype=np.float64),
            "eta": np.asarray(swampe_cont.read_pickle(f"eta-{timestamp}", custompath=custompath), dtype=np.float64),
            "delta": np.asarray(swampe_cont.read_pickle(f"delta-{timestamp}", custompath=custompath), dtype=np.float64),
            "terminal_step": int(terminal_step),
            "terminal_timestamp_seconds": int(last_ts_s),
            "target_step": int(target_step),
            "reached_target_step": bool(terminal_step >= target_step),
        }


def _run_my_swamp(*, tmax: int) -> Dict[str, np.ndarray]:
    """Run MY_SWAMP with matched settings and return terminal fields."""
    ensure_my_swamp_importable(PROJECT_ROOT)
    from my_swamp.model import run_model_scan_final

    out = run_model_scan_final(
        M=int(M),
        dt=float(DT_SECONDS),
        tmax=int(tmax),
        Phibar=float(PARAMS.Phibar),
        omega=float(PARAMS.omega_rad_s),
        a=float(PARAMS.a_m),
        test=TEST_CASE,
        g=float(PARAMS.g_m_s2),
        forcflag=FORCFLAG,
        taurad=float(PARAMS.taurad_s),
        taudrag=float(PARAMS.taudrag_s),
        DPhieq=float(PARAMS.DPhieq),
        diffflag=DIFFFLAG,
        modalflag=MODALFLAG,
        alpha=float(ALPHA),
        expflag=EXPFLAG,
        K6=float(PARAMS.K6),
        K6Phi=PARAMS.K6Phi,
        diagnostics=False,
        starttime=int(STARTTIME_INDEX),
        jit_scan=JIT_SCAN,
    )
    last = out["last_state"]
    return {
        "Phi": np.asarray(last.Phi_curr, dtype=np.float64),
        "U": np.asarray(last.U_curr, dtype=np.float64),
        "V": np.asarray(last.V_curr, dtype=np.float64),
        "eta": np.asarray(last.eta_curr, dtype=np.float64),
        "delta": np.asarray(last.delta_curr, dtype=np.float64),
    }


def _metric_dict(*, swampe_field: np.ndarray, my_swamp_field: np.ndarray, atol: float) -> Dict[str, float | bool]:
    """Compute per-field parity metrics and tolerance pass/fail flag."""
    diff = my_swamp_field - swampe_field
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    mean_abs = float(np.mean(np.abs(diff)))
    l2_ref = float(np.linalg.norm(swampe_field.ravel()))
    l2_diff = float(np.linalg.norm(diff.ravel()))
    rel_l2 = l2_diff / max(l2_ref, np.finfo(np.float64).tiny)
    return {
        "atol": float(atol),
        "max_abs": max_abs,
        "rmse": rmse,
        "mean_abs": mean_abs,
        "rel_l2": float(rel_l2),
        "allclose_rtol0": bool(np.allclose(my_swamp_field, swampe_field, rtol=0.0, atol=float(atol))),
    }


def _current_jax_backend() -> str:
    """Return active JAX backend name for diagnostic reporting."""
    try:
        import jax
    except Exception:
        return "unknown"
    try:
        return str(jax.default_backend()).lower()
    except Exception:
        return "unknown"


def main() -> None:
    """Execute terminal-state parity run and emit report + figure artifacts."""
    _apply_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if TIME_DAYS <= 0:
        raise ValueError("TIME_DAYS must be > 0")
    if DT_SECONDS <= 0:
        raise ValueError("DT_SECONDS must be > 0")
    if STARTTIME_INDEX < 2:
        raise ValueError("STARTTIME_INDEX must be >= 2 for leapfrog startup parity")

    n_steps = max(1, int(round(TIME_DAYS * 86400.0 / DT_SECONDS)))
    tmax = int(STARTTIME_INDEX + n_steps)

    fields = ["Phi", "U", "V", "eta", "delta"]
    swampe_run = _run_swampe(tmax=tmax)
    swampe_terminal_step = int(swampe_run["terminal_step"])
    compared_steps = int(swampe_terminal_step - STARTTIME_INDEX + 1)
    if compared_steps < 1:
        raise RuntimeError(
            f"Invalid terminal step from SWAMPE ({swampe_terminal_step}); expected at least STARTTIME_INDEX={STARTTIME_INDEX}."
        )
    compared_time_days = float(compared_steps * DT_SECONDS / 86400.0)
    compared_tmax = int(swampe_terminal_step + 1)
    my_swamp = _run_my_swamp(tmax=compared_tmax)
    swampe = {field: np.asarray(swampe_run[field], dtype=np.float64) for field in fields}

    metrics: Dict[str, Dict[str, float | bool]] = {}
    failed: list[str] = []
    for field in fields:
        if swampe[field].shape != my_swamp[field].shape:
            raise ValueError(
                f"Shape mismatch for {field}: SWAMPE={swampe[field].shape} vs MY_SWAMP={my_swamp[field].shape}"
            )
        m = _metric_dict(
            swampe_field=swampe[field],
            my_swamp_field=my_swamp[field],
            atol=float(ATOL_BY_FIELD[field]),
        )
        metrics[field] = m
        if not bool(m["allclose_rtol0"]):
            failed.append(field)

    fig_path = (OUT_DIR / FIGURE_NAME).resolve()
    _save_phi_figure(
        swampe_phi=swampe["Phi"],
        my_swamp_phi=my_swamp["Phi"],
        out_path=fig_path,
        steps=compared_steps,
        compared_time_days=compared_time_days,
    )

    report = {
        "comparison": "SWAMPE vs MY_SWAMP terminal state parity",
        "cpu_only": True,
        "jax_backend": _current_jax_backend(),
        "solver": {
            "M": int(M),
            "dt_seconds": float(DT_SECONDS),
            "requested_time_days": float(TIME_DAYS),
            "requested_steps": int(n_steps),
            "starttime_index": int(STARTTIME_INDEX),
            "requested_tmax": int(tmax),
            "swampe_terminal_step": int(swampe_terminal_step),
            "swampe_terminal_timestamp_seconds": int(swampe_run["terminal_timestamp_seconds"]),
            "swampe_reached_requested_terminal_step": bool(swampe_run["reached_target_step"]),
            "compared_steps": int(compared_steps),
            "compared_time_days": float(compared_time_days),
            "compared_tmax": int(compared_tmax),
            "forcflag": bool(FORCFLAG),
            "diffflag": bool(DIFFFLAG),
            "modalflag": bool(MODALFLAG),
            "expflag": bool(EXPFLAG),
            "test": TEST_CASE,
            "jit_scan": bool(JIT_SCAN),
        },
        "params": {
            "a_m": float(PARAMS.a_m),
            "omega_rad_s": float(PARAMS.omega_rad_s),
            "Phibar": float(PARAMS.Phibar),
            "DPhieq": float(PARAMS.DPhieq),
            "taurad_s": float(PARAMS.taurad_s),
            "taudrag_s": float(PARAMS.taudrag_s),
            "g_m_s2": float(PARAMS.g_m_s2),
            "K6": float(PARAMS.K6),
            "K6Phi": PARAMS.K6Phi,
            "alpha": float(ALPHA),
        },
        "metrics": metrics,
        "all_fields_allclose": len(failed) == 0,
        "failed_fields": failed,
        "figure_path": str(fig_path),
    }

    report_path = (OUT_DIR / REPORT_NAME).resolve()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {report_path}")
    print(f"Compared horizon: {compared_time_days:.6f} days ({compared_steps} steps)")
    if not bool(swampe_run["reached_target_step"]):
        print(
            "WARNING: SWAMPE did not reach the requested horizon. "
            f"Requested steps={n_steps}, reached terminal step={swampe_terminal_step}."
        )
    for field in fields:
        m = metrics[field]
        print(
            f"{field:>5s} | max_abs={float(m['max_abs']):.3e} | rmse={float(m['rmse']):.3e} "
            f"| rel_l2={float(m['rel_l2']):.3e} | pass={bool(m['allclose_rtol0'])}"
        )

    if failed and FAIL_ON_TOLERANCE:
        raise RuntimeError(f"Parity check failed for field(s): {failed}. See {report_path}")


if __name__ == "__main__":
    main()
