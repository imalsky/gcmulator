"""Integration helpers for MY_SWAMP runtime execution and state extraction.

The emulator never trains directly on the full internal MY_SWAMP carry, but it
does need a reproducible way to extract visible states and to reconstruct winds
from prognostic channels during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from config import (
    CONDITIONING_PARAM_NAMES,
    Extended9Params,
    PHYSICAL_STATE_FIELDS,
    PROGNOSTIC_TARGET_FIELDS,
    SECONDS_PER_DAY,
)


MIN_ROLLOUT_STEPS = 1
CHUNK_STEPS = 256
CURRENT_FIELD_INDICES = (0, 1, 2, 3, 4)
PROGNOSTIC_FIELD_INDICES = (0, 3, 4)


@dataclass(frozen=True)
class ReducedCarrySnapshot:
    """Minimal MY_SWAMP carry stored internally for extracting visible states."""

    Phi_curr: np.ndarray
    U_curr: np.ndarray
    V_curr: np.ndarray
    eta_curr: np.ndarray
    delta_curr: np.ndarray
    Phi_prev: np.ndarray
    eta_prev: np.ndarray
    delta_prev: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return the reduced carry stacked as ``[8,H,W]``."""
        return np.stack(
            [
                np.asarray(self.Phi_curr, dtype=np.float64),
                np.asarray(self.U_curr, dtype=np.float64),
                np.asarray(self.V_curr, dtype=np.float64),
                np.asarray(self.eta_curr, dtype=np.float64),
                np.asarray(self.delta_curr, dtype=np.float64),
                np.asarray(self.Phi_prev, dtype=np.float64),
                np.asarray(self.eta_prev, dtype=np.float64),
                np.asarray(self.delta_prev, dtype=np.float64),
            ],
            axis=0,
        )


def enforce_no_tpu_backend() -> None:
    """Force JAX backend selection to exclude TPU and keep parity-grade defaults."""
    os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    raw_platforms = os.environ.get("JAX_PLATFORMS", "")
    if raw_platforms.strip():
        parts = [part.strip() for part in raw_platforms.split(",") if part.strip()]
        kept = [part for part in parts if part.lower() != "tpu"]
        if kept:
            os.environ["JAX_PLATFORMS"] = ",".join(kept)
        else:
            os.environ.pop("JAX_PLATFORMS", None)
    else:
        os.environ.pop("JAX_PLATFORMS", None)

    if os.environ.get("JAX_PLATFORM_NAME", "").strip().lower() == "tpu":
        os.environ.pop("JAX_PLATFORM_NAME", None)


def detect_jax_backend() -> str:
    """Return the active JAX backend name when available."""
    try:
        import jax
    except Exception:
        return "unknown"
    try:
        return str(jax.default_backend()).lower()
    except Exception:
        return "unknown"


def ensure_my_swamp_importable(_: Path | None = None) -> None:
    """Require an importable ``my_swamp`` installation."""
    enforce_no_tpu_backend()
    try:
        import my_swamp  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Could not import my_swamp. Install it into the active environment "
            "first."
        ) from exc


def _snapshot_from_last_state(last_state: object) -> ReducedCarrySnapshot:
    """Convert a MY_SWAMP scan carry into a reduced carry snapshot."""
    return ReducedCarrySnapshot(
        Phi_curr=np.asarray(last_state.Phi_curr, dtype=np.float64),
        U_curr=np.asarray(last_state.U_curr, dtype=np.float64),
        V_curr=np.asarray(last_state.V_curr, dtype=np.float64),
        eta_curr=np.asarray(last_state.eta_curr, dtype=np.float64),
        delta_curr=np.asarray(last_state.delta_curr, dtype=np.float64),
        Phi_prev=np.asarray(last_state.Phi_prev, dtype=np.float64),
        eta_prev=np.asarray(last_state.eta_prev, dtype=np.float64),
        delta_prev=np.asarray(last_state.delta_prev, dtype=np.float64),
    )


def _stack_reduced_carry_state_jax(state: object) -> Any:
    """Pack the reduced carry fields into one stacked JAX tensor."""
    import jax.numpy as jnp

    return jnp.stack(
        [
            state.Phi_curr,
            state.U_curr,
            state.V_curr,
            state.eta_curr,
            state.delta_curr,
            state.Phi_prev,
            state.eta_prev,
            state.delta_prev,
        ],
        axis=0,
    )


def _build_run_flags(*, diagnostics: bool) -> Any:
    """Build the fixed MY_SWAMP runtime flags used by the emulator pipeline."""
    from my_swamp.model import RunFlags

    return RunFlags(
        forcflag=True,
        diffflag=True,
        expflag=False,
        modalflag=True,
        diagnostics=bool(diagnostics),
        alpha=0.01,
    )


def _total_rollout_steps(*, time_days: float, dt_seconds: float) -> int:
    """Return the total number of simulated steps for one trajectory run."""
    return max(
        MIN_ROLLOUT_STEPS,
        int(round(float(time_days) * SECONDS_PER_DAY / float(dt_seconds))),
    )


def _initialize_trajectory_state(
    params: Extended9Params,
    *,
    M: int,
    dt_seconds: float,
    starttime_index: int,
) -> tuple[Any, Any, Any, Any]:
    """Build static operators and the initial two-level MY_SWAMP state."""
    import jax.numpy as jnp
    from my_swamp.model import run_model_scan

    init_out = run_model_scan(
        M=int(M),
        dt=float(dt_seconds),
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
        return_history=False,
        starttime=int(starttime_index),
        tmax=int(starttime_index),
        jit_scan=True,
        donate_state=True,
    )
    current_state_full = init_out["last_state"]
    return (
        init_out["static"],
        current_state_full,
        jnp.asarray(current_state_full.U_curr),
        jnp.asarray(current_state_full.V_curr),
    )


@lru_cache(maxsize=1)
def _get_reduced_carry_chunk_runner() -> Any:
    """Build and cache a jitted chunk runner returning reduced-carry outputs."""
    import jax
    from my_swamp.model import _step_once

    def _scan_chunk(
        static: Any,
        flags: Any,
        state0: Any,
        t_seq: Any,
        Uic: Any,
        Vic: Any,
    ) -> tuple[Any, Any]:
        """Advance one chunk and collect visible states after each step."""
        import jax.numpy as jnp  # noqa: F811

        def _step(carry: Any, t: Any) -> tuple[Any, jnp.ndarray]:
            """Advance a single MY_SWAMP step inside the chunk scan."""
            new_state, _ = _step_once(carry, t, static, flags, None, Uic, Vic)
            return new_state, _stack_reduced_carry_state_jax(new_state)

        return jax.lax.scan(_step, state0, t_seq)

    return jax.jit(_scan_chunk, donate_argnums=(2,))


def run_trajectory_window(
    params: Extended9Params,
    *,
    M: int,
    dt_seconds: float,
    time_days: float,
    starttime_index: int,
    anchor_steps: np.ndarray,
    target_steps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract visible states at arbitrary (anchor, target) checkpoint pairs.

    The simulation is run for the full ``time_days`` duration and states are
    collected at the union of requested anchor and target steps using a
    memory-efficient chunked scan.  Each (anchor, target) pair may use a
    different time gap, enabling variable-dt training data.

    Args:
        params: Physical parameters for the MY_SWAMP run.
        M: Spectral truncation order.
        dt_seconds: Solver time step in seconds.
        time_days: Total simulation duration in days.
        starttime_index: Number of warm-up steps for two-level scheme.
        anchor_steps: (T,) int64 array of input-state step indices.
        target_steps: (T,) int64 array of target-state step indices.

    Returns:
        state_inputs: (T, 5, H, W) float64 — full visible state at anchors.
        state_targets: (T, 3, H, W) float64 — prognostic state at targets.
    """
    import jax.numpy as jnp

    anchor_steps = np.asarray(anchor_steps, dtype=np.int64)
    target_steps = np.asarray(target_steps, dtype=np.int64)

    if anchor_steps.ndim != 1 or target_steps.ndim != 1:
        raise ValueError("anchor_steps and target_steps must be 1-D arrays")
    if anchor_steps.shape[0] != target_steps.shape[0]:
        raise ValueError("anchor_steps and target_steps must have the same length")
    if anchor_steps.shape[0] < 1:
        raise ValueError("At least one transition pair is required")
    if time_days <= 0:
        raise ValueError("time_days must be > 0")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0")
    if np.any(anchor_steps < 0):
        raise ValueError("anchor_steps must be >= 0")
    if np.any(target_steps <= anchor_steps):
        raise ValueError("Each target_step must be > its anchor_step")

    n_steps_total = _total_rollout_steps(time_days=time_days, dt_seconds=dt_seconds)
    if int(np.max(target_steps)) > n_steps_total:
        raise ValueError(
            "Requested target step exceeds the simulated horizon: "
            f"max_target_step={int(np.max(target_steps))}, "
            f"available={n_steps_total}"
        )

    required_steps = np.unique(
        np.concatenate(
            [np.asarray([0], dtype=np.int64), anchor_steps, target_steps],
            axis=0,
        )
    )

    static, current_state_full, Uic, Vic = _initialize_trajectory_state(
        params,
        M=int(M),
        dt_seconds=float(dt_seconds),
        starttime_index=int(starttime_index),
    )
    flags = _build_run_flags(diagnostics=False)

    states_by_step: Dict[int, np.ndarray] = {
        0: _snapshot_from_last_state(current_state_full).as_array().astype(
            np.float64, copy=False
        )
    }
    required_list = [int(value) for value in required_steps.tolist()]
    req_ptr = 1
    step_cursor = 0
    chunk_runner = _get_reduced_carry_chunk_runner()

    while req_ptr < len(required_list):
        if step_cursor >= n_steps_total:
            raise RuntimeError(
                "Failed to collect all requested trajectory checkpoints before "
                "horizon end"
            )
        chunk_len = min(CHUNK_STEPS, n_steps_total - step_cursor)
        abs_t0 = starttime_index + step_cursor
        abs_t1 = abs_t0 + chunk_len
        t_seq = jnp.arange(abs_t0, abs_t1, dtype=jnp.int32)
        current_state_full, chunk_history = chunk_runner(
            static,
            flags,
            current_state_full,
            t_seq,
            Uic,
            Vic,
        )
        chunk_history_np = np.asarray(chunk_history, dtype=np.float64)
        chunk_end = step_cursor + chunk_len

        # ``chunk_history`` stores the state *after* each simulated step, so
        # the requested checkpoint at absolute step ``req_step`` lives at
        # relative index ``rel - 1``.
        while req_ptr < len(required_list) and required_list[req_ptr] <= chunk_end:
            req_step = required_list[req_ptr]
            rel = req_step - step_cursor
            if rel < 1:
                raise RuntimeError(
                    f"Invalid relative checkpoint offset {rel} "
                    f"for req_step={req_step}"
                )
            states_by_step[req_step] = chunk_history_np[rel - 1].astype(
                np.float64, copy=False
            )
            req_ptr += 1

        step_cursor = chunk_end

    current_field_indices = list(CURRENT_FIELD_INDICES)
    prognostic_field_indices = list(PROGNOSTIC_FIELD_INDICES)

    state_inputs = np.stack(
        [
            np.take(states_by_step[int(step)], current_field_indices, axis=0)
            for step in anchor_steps.tolist()
        ],
        axis=0,
    ).astype(np.float64)
    state_targets = np.stack(
        [
            np.take(states_by_step[int(step)], prognostic_field_indices, axis=0)
            for step in target_steps.tolist()
        ],
        axis=0,
    ).astype(np.float64)
    return state_inputs, state_targets


@lru_cache(maxsize=128)
def _get_diagnostic_static(
    *,
    M: int,
    dt_seconds: float,
    a_m: float,
    omega_rad_s: float,
    Phibar: float,
    DPhieq: float,
    taurad_s: float,
    taudrag_s: float,
    g_m_s2: float,
    K6: float,
    K6Phi: float | None,
) -> Any:
    """Cache MY_SWAMP static spectral operators for deterministic wind diagnosis."""
    from my_swamp.model import build_static

    return build_static(
        M=int(M),
        dt=float(dt_seconds),
        a=float(a_m),
        omega=float(omega_rad_s),
        g=float(g_m_s2),
        Phibar=float(Phibar),
        taurad=float(taurad_s),
        taudrag=float(taudrag_s),
        DPhieq=float(DPhieq),
        K6=float(K6),
        K6Phi=K6Phi,
        test=None,
    )


def diagnose_winds(
    eta: np.ndarray,
    delta: np.ndarray,
    *,
    params: Extended9Params,
    M: int,
    dt_seconds: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagnose physical-space ``U,V`` from physical-space ``eta,delta``."""
    import jax.numpy as jnp
    from my_swamp import spectral_transform as st

    static = _get_diagnostic_static(
        M=int(M),
        dt_seconds=float(dt_seconds),
        a_m=float(params.a_m),
        omega_rad_s=float(params.omega_rad_s),
        Phibar=float(params.Phibar),
        DPhieq=float(params.DPhieq),
        taurad_s=float(params.taurad_s),
        taudrag_s=float(params.taudrag_s),
        g_m_s2=float(params.g_m_s2),
        K6=float(params.K6),
        K6Phi=params.K6Phi,
    )
    eta_j = jnp.asarray(eta)
    delta_j = jnp.asarray(delta)
    etam, deltam = st.fwd_fft_trunc_batch(
        jnp.stack((eta_j, delta_j), axis=0), static.I, static.M
    )
    etamn = st.fwd_leg(etam, static.J, static.M, static.N, static.Pmn, static.w)
    deltamn = st.fwd_leg(deltam, static.J, static.M, static.N, static.Pmn, static.w)
    u_complex, v_complex = st.invrsUV(
        deltamn,
        etamn,
        static.fmn,
        static.I,
        static.J,
        static.M,
        static.N,
        static.Pmn,
        static.Hmn,
        static.tstepcoeffmn,
        static.marray,
    )
    return (
        np.asarray(jnp.real(u_complex), dtype=np.float64),
        np.asarray(jnp.real(v_complex), dtype=np.float64),
    )


def reconstruct_full_state_from_prognostics(
    prognostics: np.ndarray,
    *,
    params: Extended9Params,
    M: int,
    dt_seconds: float,
) -> np.ndarray:
    """Reconstruct a full physical 5-field state from prognostic ``Phi,eta,delta``."""
    if prognostics.shape[0] != len(PROGNOSTIC_TARGET_FIELDS):
        raise ValueError(
            "prognostics must have "
            f"{len(PROGNOSTIC_TARGET_FIELDS)} channels, "
            f"got {prognostics.shape[0]}"
        )
    phi = np.asarray(prognostics[0], dtype=np.float64)
    eta = np.asarray(prognostics[1], dtype=np.float64)
    delta = np.asarray(prognostics[2], dtype=np.float64)
    u_field, v_field = diagnose_winds(
        eta, delta, params=params, M=M, dt_seconds=dt_seconds
    )
    return np.stack([phi, u_field, v_field, eta, delta], axis=0)


def conditioning_param_names() -> Tuple[str, ...]:
    """Return canonical user-facing conditioning parameter ordering."""
    return tuple(CONDITIONING_PARAM_NAMES)


def params_to_conditioning_vector(params: Extended9Params) -> np.ndarray:
    """Return the conditioning vector used by the ML model."""
    return np.asarray(params.to_vector(), dtype=np.float64)


def params_to_public_json_dict(params: Extended9Params) -> Dict[str, float]:
    """Return a JSON-friendly user-facing parameter dictionary."""
    return {
        "a_m": float(params.a_m),
        "omega_rad_s": float(params.omega_rad_s),
        "Phibar": float(params.Phibar),
        "DPhieq": float(params.DPhieq),
        "taurad_s": float(params.taurad_s),
        "taudrag_s": float(params.taudrag_s),
        "g_m_s2": float(params.g_m_s2),
        "K6": float(params.K6),
        "K6Phi": params.K6Phi,
    }
