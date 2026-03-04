from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .config import Extended9Params, TerminalState
from .constants import CANONICAL_VORTICITY_FACTOR, MIN_ROLLOUT_STEPS, SECONDS_PER_DAY

FIELDS_5 = ["Phi", "U", "V", "eta", "delta"]


def enforce_no_tpu_backend() -> None:
    """Force JAX backend selection to exclude TPU.

    Keeps existing user selection when possible, but removes `tpu`.
    Falls back to JAX auto-selection when no explicit non-TPU platform remains.
    """
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


def _build_static(params: Extended9Params, *, M: int, dt_seconds: float):
    from my_swamp.model import build_static
    import jax.numpy as jnp

    k6phi = None if params.K6Phi is None else jnp.asarray(float(params.K6Phi), dtype=jnp.float64)

    return build_static(
        M=int(M),
        dt=jnp.asarray(float(dt_seconds), dtype=jnp.float64),
        a=jnp.asarray(float(params.a_m), dtype=jnp.float64),
        omega=jnp.asarray(float(params.omega_rad_s), dtype=jnp.float64),
        g=jnp.asarray(float(params.g_m_s2), dtype=jnp.float64),
        Phibar=jnp.asarray(float(params.Phibar), dtype=jnp.float64),
        taurad=jnp.asarray(float(params.taurad_s), dtype=jnp.float64),
        taudrag=jnp.asarray(float(params.taudrag_s), dtype=jnp.float64),
        DPhieq=jnp.asarray(float(params.DPhieq), dtype=jnp.float64),
        K6=jnp.asarray(float(params.K6), dtype=jnp.float64),
        K6Phi=k6phi,
        test=None,
    )


def build_canonical_initial_state(
    params: Extended9Params,
    *,
    M: int,
    dt_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic canonical IC based on MY_SWAMP static state.

    Mirrors the rest-state pattern used in MY_SWAMP retrieval script.
    Returns (eta0, delta0, Phi0, U0, V0) each [J,I].
    """
    static = _build_static(params, M=M, dt_seconds=dt_seconds)

    mus = np.asarray(static.mus, dtype=np.float64)
    omega = float(np.asarray(static.omega, dtype=np.float64))
    J = int(getattr(static, "J"))
    I = int(getattr(static, "I"))

    eta1d = CANONICAL_VORTICITY_FACTOR * omega * mus
    eta0 = eta1d[:, None] * np.ones((J, I), dtype=np.float64)
    delta0 = np.zeros((J, I), dtype=np.float64)
    U0 = np.zeros((J, I), dtype=np.float64)
    V0 = np.zeros((J, I), dtype=np.float64)

    phieq = np.asarray(static.Phieq, dtype=np.float64)
    phibar = float(np.asarray(static.Phibar, dtype=np.float64))
    phi0 = phieq - phibar

    return eta0, delta0, phi0, U0, V0


def run_terminal_state(
    params: Extended9Params,
    *,
    M: int,
    dt_seconds: float,
    time_days: float,
    starttime_index: int,
) -> TerminalState:
    """Run MY_SWAMP to terminal time and return full 5-field state."""
    from my_swamp.model import run_model_scan_final

    if time_days <= 0:
        raise ValueError("time_days must be > 0")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0")

    n_steps = int(round(float(time_days) * SECONDS_PER_DAY / float(dt_seconds)))
    n_steps = max(MIN_ROLLOUT_STEPS, n_steps)
    tmax = int(starttime_index + n_steps)

    eta0, delta0, phi0, u0, v0 = build_canonical_initial_state(params, M=M, dt_seconds=dt_seconds)

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
        eta0_init=eta0,
        delta0_init=delta0,
        Phi0_init=phi0,
        U0_init=u0,
        V0_init=v0,
        jit_scan=True,
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


def param_names_extended9() -> Tuple[str, ...]:
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
    return params.to_vector().astype(np.float64)


def params_to_json_dict(params: Extended9Params) -> Dict[str, float | None]:
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
