"""Integration helpers for MY_SWAMP runtime execution and metadata conversion."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import Extended9Params, TerminalState

# Integration timing constants.
MIN_ROLLOUT_STEPS = 1
SECONDS_PER_DAY = 86400.0

FIELDS_5 = ["Phi", "U", "V", "eta", "delta"]


def enforce_no_tpu_backend() -> None:
    """Force JAX backend selection to exclude TPU and set safe runtime defaults.

    Keeps existing user selection when possible, but removes `tpu`.
    Falls back to JAX auto-selection when no explicit non-TPU platform remains.
    Also defaults MY_SWAMP generation to float32 and disables aggressive
    XLA preallocation unless the user has explicitly configured these.
    """
    os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    raw_platforms = os.environ.get("JAX_PLATFORMS", "")
    if raw_platforms.strip():
        parts = [p.strip() for p in raw_platforms.split(",") if p.strip()]
        kept = [p for p in parts if p.lower() != "tpu"]
        if kept:
            os.environ["JAX_PLATFORMS"] = ",".join(kept)
        else:
            os.environ.pop("JAX_PLATFORMS", None)
    else:
        os.environ.pop("JAX_PLATFORMS", None)

    if os.environ.get("JAX_PLATFORM_NAME", "").strip().lower() == "tpu":
        os.environ.pop("JAX_PLATFORM_NAME", None)


def detect_jax_backend() -> str:
    """Return active JAX backend name when available."""
    try:
        import jax  # type: ignore
    except Exception:
        return "unknown"
    try:
        return str(jax.default_backend()).lower()
    except Exception:
        return "unknown"


def ensure_my_swamp_importable(config_dir: Path) -> None:
    """Ensure `my_swamp` can be imported.

    Preference order:
    1) installed package
    2) sibling checkout (`MY_SWAMP`)
    """
    enforce_no_tpu_backend()

    try:
        import my_swamp  # noqa: F401
        return
    except Exception:
        pass

    candidates = [
        config_dir / "MY_SWAMP" / "src",
        config_dir.parent / "MY_SWAMP" / "src",
    ]
    for c in candidates:
        if c.is_dir():
            os.sys.path.insert(0, str(c))
            try:
                import my_swamp  # noqa: F401
                return
            except Exception:
                os.sys.path.pop(0)

    raise RuntimeError("Could not import my_swamp. Install it or place MY_SWAMP/src near this config.")


def run_terminal_state(
    params: Extended9Params,
    *,
    M: int,
    dt_seconds: float,
    time_days: float,
    starttime_index: int,
) -> TerminalState:
    """Run MY_SWAMP to terminal time and return full 5-field state.

    Uses MY_SWAMP/SWAMPE built-in analytic initialization (`test=None`) with the
    model's native two-level startup rather than overriding explicit IC fields.
    """
    from my_swamp.model import run_model_scan_final

    if time_days <= 0:
        raise ValueError("time_days must be > 0")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0")

    n_steps = int(round(float(time_days) * SECONDS_PER_DAY / float(dt_seconds)))
    n_steps = max(MIN_ROLLOUT_STEPS, n_steps)
    tmax = int(starttime_index + n_steps)

    out = run_model_scan_final(
        M=int(M),
        dt=float(dt_seconds),
        tmax=int(tmax),
        Phibar=float(params.Phibar),
        omega=float(params.omega_rad_s),
        a=float(params.a_m),
        test=None,
        g=float(params.g_m_s2),
        forcflag=True,
        taurad=float(params.taurad_s),
        taudrag=float(params.taudrag_s),
        DPhieq=float(params.DPhieq),
        diffflag=True,
        modalflag=True,
        expflag=False,
        K6=float(params.K6),
        K6Phi=params.K6Phi,
        diagnostics=False,
        starttime=int(starttime_index),
        jit_scan=True,
        donate_state=True,
    )

    ls = out["last_state"]
    state = TerminalState(
        phi=np.asarray(ls.Phi_curr, dtype=np.float32),
        u=np.asarray(ls.U_curr, dtype=np.float32),
        v=np.asarray(ls.V_curr, dtype=np.float32),
        eta=np.asarray(ls.eta_curr, dtype=np.float32),
        delta=np.asarray(ls.delta_curr, dtype=np.float32),
    )
    return state


@lru_cache(maxsize=None)
def _get_batched_terminal_runner(
    *,
    M: int,
    dt_seconds: float,
    tmax: int,
    starttime_index: int,
):
    """Build and cache a vmapped terminal-state runner for fixed solver settings."""
    import jax
    import jax.numpy as jnp
    from my_swamp.model import run_model_scan_final

    def _one(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        out = run_model_scan_final(
            M=int(M),
            dt=float(dt_seconds),
            tmax=int(tmax),
            Phibar=theta[2],
            omega=theta[1],
            a=theta[0],
            test=None,
            g=theta[6],
            forcflag=True,
            taurad=theta[4],
            taudrag=theta[5],
            DPhieq=theta[3],
            diffflag=True,
            modalflag=True,
            expflag=False,
            K6=theta[7],
            K6Phi=theta[8],
            diagnostics=False,
            starttime=int(starttime_index),
            jit_scan=True,
            donate_state=True,
        )
        ls = out["last_state"]
        return ls.Phi_curr, ls.U_curr, ls.V_curr, ls.eta_curr, ls.delta_curr

    return jax.vmap(_one, in_axes=0, out_axes=0)


def run_terminal_state_batch(
    params_batch: List[Extended9Params],
    *,
    M: int,
    dt_seconds: float,
    time_days: float,
    starttime_index: int,
) -> List[TerminalState]:
    """Run MY_SWAMP for a batch of independent parameter sets using JAX vmap."""
    if not params_batch:
        return []
    if time_days <= 0:
        raise ValueError("time_days must be > 0")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0")

    n_steps = int(round(float(time_days) * SECONDS_PER_DAY / float(dt_seconds)))
    n_steps = max(MIN_ROLLOUT_STEPS, n_steps)
    tmax = int(starttime_index + n_steps)

    # [a, omega, Phibar, DPhieq, taurad, taudrag, g, K6, K6Phi_eff]
    # For compatibility with scalar path semantics, map K6Phi=None -> K6.
    theta_rows: List[List[float]] = []
    for p in params_batch:
        k6phi_eff = float(p.K6) if p.K6Phi is None else float(p.K6Phi)
        theta_rows.append(
            [
                float(p.a_m),
                float(p.omega_rad_s),
                float(p.Phibar),
                float(p.DPhieq),
                float(p.taurad_s),
                float(p.taudrag_s),
                float(p.g_m_s2),
                float(p.K6),
                k6phi_eff,
            ]
        )
    theta = np.asarray(theta_rows, dtype=np.float32)

    runner = _get_batched_terminal_runner(
        M=int(M),
        dt_seconds=float(dt_seconds),
        tmax=int(tmax),
        starttime_index=int(starttime_index),
    )

    phi_b, u_b, v_b, eta_b, delta_b = runner(theta)
    states = []
    for i in range(len(params_batch)):
        states.append(
            TerminalState(
                phi=np.asarray(phi_b[i], dtype=np.float32),
                u=np.asarray(u_b[i], dtype=np.float32),
                v=np.asarray(v_b[i], dtype=np.float32),
                eta=np.asarray(eta_b[i], dtype=np.float32),
                delta=np.asarray(delta_b[i], dtype=np.float32),
            )
        )
    return states


def param_names_extended9() -> Tuple[str, ...]:
    """Return canonical parameter-name ordering used in stored arrays."""
    return (
        "a_m",
        "omega_rad_s",
        "Phibar",
        "DPhieq",
        "taurad_s",
        "taudrag_s",
        "g_m_s2",
        "K6",
        "K6Phi",
    )


def params_to_ordered_vector(params: Extended9Params) -> np.ndarray:
    """Serialize dataclass parameters to canonical float64 vector order."""
    return params.to_vector().astype(np.float64)


def params_to_json_dict(params: Extended9Params) -> Dict[str, float | None]:
    """Serialize parameters to JSON-friendly name-value mapping."""
    return {
        "a_m": params.a_m,
        "omega_rad_s": params.omega_rad_s,
        "Phibar": params.Phibar,
        "DPhieq": params.DPhieq,
        "taurad_s": params.taurad_s,
        "taudrag_s": params.taudrag_s,
        "g_m_s2": params.g_m_s2,
        "K6": params.K6,
        "K6Phi": params.K6Phi,
    }
