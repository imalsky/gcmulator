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

from .config import (
    CONDITIONING_PARAM_NAMES,
    Extended9Params,
    PHYSICAL_STATE_FIELDS,
    PROGNOSTIC_STATE_FIELDS,
    SECONDS_PER_DAY,
    VALID_CONVECTIVE_FORCING_MODES,
    VALID_INITIAL_CONDITION_MODES,
)


MIN_ROLLOUT_STEPS = 1
# Number of solver time-steps advanced per jitted scan chunk. Balances JIT
# compilation overhead against peak memory from the stacked chunk history.
CHUNK_STEPS = 256
# Indices into the 8-field reduced carry that select the visible physical
# state: Phi_curr(0), U_curr(1), V_curr(2), eta_curr(3), delta_curr(4).
CURRENT_FIELD_INDICES = (0, 1, 2, 3, 4)
STORM_TIME_WINDOW_SIGMAS = 3.0
_UINT64_HASH_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
_UINT64_HASH_MULT1 = np.uint64(0xBF58476D1CE4E5B9)
_UINT64_HASH_MULT2 = np.uint64(0x94D049BB133111EB)
_UINT64_HASH_MASK_53 = np.uint64((1 << 53) - 1)
_STORM_LON_SALT = np.uint64(0x243F6A8885A308D3)
_STORM_MU_SALT = np.uint64(0x13198A2E03707344)


def _ensure_numpy_asarray_copy_compat() -> None:
    """Patch NumPy 1.x ``asarray`` so JAX can pass the newer ``copy=`` kwarg."""
    try:
        np.asarray(0, copy=None)  # type: ignore[call-arg]
        return
    except TypeError:
        pass

    original_asarray = np.asarray

    def _compat_asarray(
        a: Any,
        dtype: Any = None,
        order: str | None = None,
        *,
        like: Any = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        if copy is None:
            return original_asarray(a, dtype=dtype, order=order, like=like)
        if copy:
            return np.array(a, dtype=dtype, order=order, like=like, copy=True)
        return original_asarray(a, dtype=dtype, order=order, like=like)

    np.asarray = _compat_asarray  # type: ignore[assignment]


_ensure_numpy_asarray_copy_compat()


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
    """Return the active JAX backend name.

    Raises if JAX is not importable or the backend cannot be determined.
    """
    import jax

    return str(jax.default_backend()).lower()


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


def _forcing_mode_enabled(forcing_mode: str) -> bool:
    """Return whether the configured solver mode applies external forcing."""
    mode = str(forcing_mode)
    if mode == "forced":
        return True
    if mode == "unforced":
        return False
    raise ValueError(
        "forcing_mode must be one of ['forced','unforced'], "
        f"got {forcing_mode!r}"
    )


def _forcing_mode_dphieq(forcing_mode: str, DPhieq: Any) -> Any:
    """Return the effective equilibrium-contrast amplitude for one solver mode."""
    return DPhieq if _forcing_mode_enabled(forcing_mode) else 0.0


def _convective_forcing_mode_enabled(convective_forcing_mode: str) -> bool:
    """Return whether the runtime should apply gcmulator-side convective forcing."""
    mode = str(convective_forcing_mode)
    if mode == "none":
        return False
    if mode == "localized_random_storms":
        return True
    raise ValueError(
        "convective_forcing_mode must be one of "
        f"{sorted(VALID_CONVECTIVE_FORCING_MODES)!r}, got {convective_forcing_mode!r}"
    )


def _initial_condition_mode_enabled(initial_condition_mode: str) -> bool:
    """Return whether the runtime should use MY_SWAMP's built-in initialization."""
    mode = str(initial_condition_mode)
    if mode == "legacy":
        return True
    if mode == "rest":
        return False
    raise ValueError(
        "initial_condition_mode must be one of "
        f"{sorted(VALID_INITIAL_CONDITION_MODES)!r}, got {initial_condition_mode!r}"
    )


def _storm_padding_count(
    *,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
) -> int:
    """Return the number of storm intervals needed to cover the Gaussian tails."""
    return max(
        1,
        int(
            np.ceil(
                STORM_TIME_WINDOW_SIGMAS
                * float(storm_nondim_lifetime)
                / float(storm_nondim_interval)
            )
        ),
    )


def _max_storm_count_for_omegas(
    *,
    n_steps_total: int,
    dt_seconds: float,
    omega_rad_s_values: np.ndarray,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
) -> int:
    """Return the padded maximum storm-table length needed for one rollout horizon."""
    omegas = np.asarray(omega_rad_s_values, dtype=np.float64)
    if omegas.size == 0:
        return 0
    pad_count = _storm_padding_count(
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
    )
    total_time_seconds = float(n_steps_total) * float(dt_seconds)
    tau_int_seconds = float(storm_nondim_interval) / (2.0 * omegas)
    counts = np.ceil(total_time_seconds / tau_int_seconds).astype(np.int64)
    counts = counts + (2 * int(pad_count)) + 1
    return int(np.max(np.maximum(counts, 1)))


def _weighted_sphere_mean_jax(field: Any, quadrature_weights: Any) -> Any:
    """Return the area-weighted sphere mean of one ``[..., J, I]`` field."""
    import jax.numpy as jnp

    weights = jnp.asarray(quadrature_weights, dtype=jnp.asarray(field).dtype)
    zonal_mean = jnp.mean(jnp.asarray(field), axis=-1)
    return jnp.sum(zonal_mean * weights, axis=-1) / jnp.sum(weights)


def _weighted_sphere_rms_jax(field: Any, quadrature_weights: Any) -> Any:
    """Return the area-weighted RMS of one ``[..., J, I]`` field."""
    import jax.numpy as jnp

    squared_mean = _weighted_sphere_mean_jax(jnp.square(jnp.asarray(field)), quadrature_weights)
    return jnp.sqrt(jnp.maximum(squared_mean, jnp.asarray(0.0, dtype=jnp.asarray(field).dtype)))


def _uniform_hash_jax(*, trajectory_seed: Any, indices: Any, salt: np.uint64, dtype: Any) -> Any:
    """Map one seed/index table to deterministic ``[0,1)`` samples."""
    import jax.numpy as jnp

    seed = jnp.asarray(trajectory_seed, dtype=jnp.uint64)
    storm_indices = jnp.asarray(indices, dtype=jnp.uint64)
    hashed = seed + jnp.asarray(salt, dtype=jnp.uint64) + storm_indices * jnp.asarray(
        _UINT64_HASH_INCREMENT, dtype=jnp.uint64
    )
    hashed = (hashed ^ (hashed >> jnp.asarray(30, dtype=jnp.uint64))) * jnp.asarray(
        _UINT64_HASH_MULT1, dtype=jnp.uint64
    )
    hashed = (hashed ^ (hashed >> jnp.asarray(27, dtype=jnp.uint64))) * jnp.asarray(
        _UINT64_HASH_MULT2, dtype=jnp.uint64
    )
    hashed = hashed ^ (hashed >> jnp.asarray(31, dtype=jnp.uint64))
    mantissa = hashed & jnp.asarray(_UINT64_HASH_MASK_53, dtype=jnp.uint64)
    return mantissa.astype(dtype) / jnp.asarray(float(1 << 53), dtype=dtype)


def _localized_storm_tendency_jax(
    *,
    lambdas: Any,
    mus: Any,
    quadrature_weights: Any,
    omega_rad_s: Any,
    Phibar: Any,
    t_seconds: Any,
    total_time_seconds: float,
    trajectory_seed: Any,
    max_storms: int,
    storm_padding_count: int,
    storm_radius_degrees: float,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
    storm_strength_fraction: float,
) -> Any:
    """Return the zero-mean storm-induced geopotential tendency in physical space."""
    import jax.numpy as jnp

    mus_j = jnp.asarray(mus)
    lambdas_j = jnp.asarray(lambdas)
    dtype = jnp.result_type(
        mus_j,
        lambdas_j,
        jnp.asarray(omega_rad_s),
        jnp.asarray(Phibar),
        jnp.asarray(t_seconds),
    )
    if int(max_storms) < 1 or float(storm_strength_fraction) == 0.0:
        return jnp.zeros((int(mus_j.shape[0]), int(lambdas_j.shape[0])), dtype=dtype)

    omega = jnp.asarray(omega_rad_s, dtype=dtype)
    tau_s_seconds = jnp.asarray(float(storm_nondim_lifetime), dtype=dtype) / (2.0 * omega)
    tau_int_seconds = jnp.asarray(float(storm_nondim_interval), dtype=dtype) / (2.0 * omega)
    total_time = jnp.asarray(float(total_time_seconds), dtype=dtype)
    storm_count = (
        jnp.ceil(total_time / tau_int_seconds).astype(jnp.int32)
        + jnp.asarray(2 * int(storm_padding_count) + 1, dtype=jnp.int32)
    )

    storm_indices = jnp.arange(int(max_storms), dtype=jnp.int32)
    valid_mask = storm_indices < storm_count
    storm_indices_f = storm_indices.astype(dtype)
    center_times = (
        storm_indices_f - jnp.asarray(int(storm_padding_count), dtype=dtype)
    ) * tau_int_seconds
    center_lons = 2.0 * jnp.pi * _uniform_hash_jax(
        trajectory_seed=trajectory_seed,
        indices=storm_indices,
        salt=_STORM_LON_SALT,
        dtype=dtype,
    )
    center_mus = 2.0 * _uniform_hash_jax(
        trajectory_seed=trajectory_seed,
        indices=storm_indices,
        salt=_STORM_MU_SALT,
        dtype=dtype,
    ) - 1.0

    mu_grid = mus_j.astype(dtype)[:, None]
    lambda_grid = jnp.mod(lambdas_j.astype(dtype), 2.0 * jnp.pi)[None, :]
    cosphi_grid = jnp.sqrt(jnp.maximum(0.0, 1.0 - jnp.square(mu_grid)))
    center_mu = center_mus[:, None, None]
    center_lon = center_lons[:, None, None]
    center_cosphi = jnp.sqrt(jnp.maximum(0.0, 1.0 - jnp.square(center_mus)))[:, None, None]
    cos_gamma = (
        mu_grid[None, :, :] * center_mu
        + cosphi_grid[None, :, :]
        * center_cosphi
        * jnp.cos(lambda_grid[None, :, :] - center_lon)
    )
    gamma = jnp.arccos(jnp.clip(cos_gamma, -1.0, 1.0))
    radius_radians = jnp.asarray(
        float(storm_radius_degrees) * (np.pi / 180.0),
        dtype=dtype,
    )
    spatial = jnp.exp(-jnp.square(gamma / radius_radians))
    temporal = jnp.exp(
        -jnp.square((jnp.asarray(t_seconds, dtype=dtype) - center_times) / tau_s_seconds)
    )[:, None, None]
    S0 = (
        jnp.asarray(float(storm_strength_fraction), dtype=dtype)
        * jnp.asarray(Phibar, dtype=dtype)
        / tau_s_seconds
    )
    storm_tendency = S0 * jnp.sum(
        spatial * temporal * valid_mask[:, None, None].astype(dtype),
        axis=0,
    )
    storm_tendency_mean = _weighted_sphere_mean_jax(storm_tendency, quadrature_weights)
    return storm_tendency - storm_tendency_mean[..., None, None]


def _build_initial_phi_noise_jax(
    *,
    lambdas: Any,
    mus: Any,
    quadrature_weights: Any,
    omega_rad_s: Any,
    Phibar: Any,
    total_time_seconds: float,
    trajectory_seed: Any,
    max_storms: int,
    storm_padding_count: int,
    initial_phi_noise_temperature_k: float,
    r_specific_j_per_kg_k: float,
    storm_radius_degrees: float,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
    storm_strength_fraction: float,
) -> Any:
    """Return a zero-mean initial ``Phi`` perturbation with the requested RMS."""
    import jax.numpy as jnp

    if float(initial_phi_noise_temperature_k) <= 0.0 or int(max_storms) < 1:
        return jnp.zeros(
            (int(jnp.asarray(mus).shape[0]), int(jnp.asarray(lambdas).shape[0])),
            dtype=jnp.float64,
        )

    base_field = _localized_storm_tendency_jax(
        lambdas=lambdas,
        mus=mus,
        quadrature_weights=quadrature_weights,
        omega_rad_s=omega_rad_s,
        Phibar=Phibar,
        t_seconds=0.0,
        total_time_seconds=float(total_time_seconds),
        trajectory_seed=trajectory_seed,
        max_storms=int(max_storms),
        storm_padding_count=int(storm_padding_count),
        storm_radius_degrees=float(storm_radius_degrees),
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
        storm_strength_fraction=float(storm_strength_fraction),
    )
    base_field = base_field - _weighted_sphere_mean_jax(
        base_field,
        quadrature_weights,
    )[..., None, None]
    rms = _weighted_sphere_rms_jax(base_field, quadrature_weights)
    target_rms = jnp.asarray(
        float(initial_phi_noise_temperature_k) * float(r_specific_j_per_kg_k),
        dtype=base_field.dtype,
    )
    scale = jnp.where(
        rms > jnp.asarray(0.0, dtype=base_field.dtype),
        target_rms / rms,
        jnp.asarray(0.0, dtype=base_field.dtype),
    )
    return base_field * scale


def _rest_initial_condition_kwargs(
    *,
    M: int,
    omega_rad_s: Any,
    Phibar: Any,
    dt_seconds: float,
    n_steps_total: int,
    trajectory_seed: Any,
    max_storms: int,
    initial_phi_noise_temperature_k: float,
    r_specific_j_per_kg_k: float,
    storm_radius_degrees: float,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
    storm_strength_fraction: float,
) -> Dict[str, Any]:
    """Build exact rest-state initial conditions for MY_SWAMP."""
    import jax.numpy as jnp
    from my_swamp import initial_conditions

    _, I, J, _, lambdas, mus, quadrature_weights = initial_conditions.spectral_params(int(M))
    omega = jnp.asarray(omega_rad_s, dtype=jnp.float64)
    mu = jnp.asarray(mus, dtype=omega.dtype)[:, None]
    eta0 = 2.0 * omega * jnp.broadcast_to(mu, (int(J), int(I)))
    zeros = jnp.zeros((int(J), int(I)), dtype=eta0.dtype)
    pad_count = _storm_padding_count(
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
    )
    Phi0 = _build_initial_phi_noise_jax(
        lambdas=lambdas,
        mus=mus,
        quadrature_weights=quadrature_weights,
        omega_rad_s=omega_rad_s,
        Phibar=Phibar,
        total_time_seconds=float(n_steps_total) * float(dt_seconds),
        trajectory_seed=trajectory_seed,
        max_storms=int(max_storms),
        storm_padding_count=int(pad_count),
        initial_phi_noise_temperature_k=float(initial_phi_noise_temperature_k),
        r_specific_j_per_kg_k=float(r_specific_j_per_kg_k),
        storm_radius_degrees=float(storm_radius_degrees),
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
        storm_strength_fraction=float(storm_strength_fraction),
    )
    return {
        "eta0_init": eta0,
        "delta0_init": zeros,
        "Phi0_init": Phi0,
        "U0_init": zeros,
        "V0_init": zeros,
    }


def _initial_condition_kwargs(
    *,
    M: int,
    Phibar: Any,
    omega_rad_s: Any,
    dt_seconds: float,
    n_steps_total: int,
    initial_condition_mode: str,
    trajectory_seed: Any,
    max_storms: int,
    initial_phi_noise_temperature_k: float,
    r_specific_j_per_kg_k: float,
    storm_radius_degrees: float,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
    storm_strength_fraction: float,
) -> Dict[str, Any]:
    """Return any explicit initial-condition overrides for one runtime mode."""
    if _initial_condition_mode_enabled(initial_condition_mode):
        return {}
    return _rest_initial_condition_kwargs(
        M=int(M),
        omega_rad_s=omega_rad_s,
        Phibar=Phibar,
        dt_seconds=float(dt_seconds),
        n_steps_total=int(n_steps_total),
        trajectory_seed=trajectory_seed,
        max_storms=int(max_storms),
        initial_phi_noise_temperature_k=float(initial_phi_noise_temperature_k),
        r_specific_j_per_kg_k=float(r_specific_j_per_kg_k),
        storm_radius_degrees=float(storm_radius_degrees),
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
        storm_strength_fraction=float(storm_strength_fraction),
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


def _recompute_phi_state_terms(*, state: Any, static: Any, Phi_prev: Any, Phi_curr: Any) -> Any:
    """Refresh the state fields that depend on the corrected ``Phi`` carry."""
    import jax.numpy as jnp
    from my_swamp import spectral_transform as st
    from my_swamp.model import _nonlinear_spectral

    Phim_prev, Phim_curr = st.fwd_fft_trunc_batch(
        jnp.stack((Phi_prev, Phi_curr), axis=0),
        static.I,
        static.M,
    )
    Am_curr, Bm_curr, Cm_curr, Dm_curr, Em_curr = _nonlinear_spectral(
        static=static,
        eta=state.eta_curr,
        Phi=Phi_curr,
        U=state.U_curr,
        V=state.V_curr,
    )
    return state._replace(
        Phi_prev=Phi_prev,
        Phi_curr=Phi_curr,
        Phim_prev=Phim_prev,
        Phim_curr=Phim_curr,
        Am_curr=Am_curr,
        Bm_curr=Bm_curr,
        Cm_curr=Cm_curr,
        Dm_curr=Dm_curr,
        Em_curr=Em_curr,
        PhiFm_curr=jnp.zeros_like(state.PhiFm_curr),
        Fm_curr=jnp.zeros_like(state.Fm_curr),
        Gm_curr=jnp.zeros_like(state.Gm_curr),
    )


def _apply_convective_phi_update(
    *,
    state: Any,
    static: Any,
    t: Any,
    starttime_index: int,
    total_time_seconds: float,
    trajectory_seed: Any,
    convective_forcing_mode: str,
    max_storms: int,
    storm_padding_count: int,
    storm_radius_degrees: float,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
    storm_strength_fraction: float,
) -> Any:
    """Apply the gcmulator-side stochastic convective ``Phi`` correction."""
    import jax.numpy as jnp

    if not _convective_forcing_mode_enabled(convective_forcing_mode):
        return state

    step_seconds = (
        jnp.asarray(t, dtype=state.Phi_curr.dtype)
        - jnp.asarray(int(starttime_index), dtype=state.Phi_curr.dtype)
        + jnp.asarray(1.0, dtype=state.Phi_curr.dtype)
    ) * jnp.asarray(static.dt, dtype=state.Phi_curr.dtype)
    storm_tendency = _localized_storm_tendency_jax(
        lambdas=static.lambdas,
        mus=static.mus,
        quadrature_weights=static.w,
        omega_rad_s=static.omega,
        Phibar=static.Phibar,
        t_seconds=step_seconds,
        total_time_seconds=float(total_time_seconds),
        trajectory_seed=trajectory_seed,
        max_storms=int(max_storms),
        storm_padding_count=int(storm_padding_count),
        storm_radius_degrees=float(storm_radius_degrees),
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
        storm_strength_fraction=float(storm_strength_fraction),
    )
    dPhi_dt = storm_tendency - state.Phi_curr / jnp.asarray(static.taurad, dtype=state.Phi_curr.dtype)
    delta_phi = jnp.asarray(static.dt, dtype=state.Phi_curr.dtype) * dPhi_dt
    return _recompute_phi_state_terms(
        state=state,
        static=static,
        Phi_prev=state.Phi_prev + delta_phi,
        Phi_curr=state.Phi_curr + delta_phi,
    )


def _build_run_flags(*, diagnostics: bool, forcing_mode: str) -> Any:
    """Build the fixed MY_SWAMP runtime flags used by the emulator pipeline."""
    from my_swamp.model import RunFlags

    return RunFlags(
        forcflag=_forcing_mode_enabled(forcing_mode),
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
    n_steps_total: int,
    starttime_index: int,
    forcing_mode: str,
    initial_condition_mode: str,
    trajectory_seed: int,
    max_storms: int,
    initial_phi_noise_temperature_k: float,
    r_specific_j_per_kg_k: float,
    storm_radius_degrees: float,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
    storm_strength_fraction: float,
) -> tuple[Any, Any, Any, Any]:
    """Build static operators and the initial two-level MY_SWAMP state."""
    import jax.numpy as jnp
    from my_swamp.model import run_model_scan

    forcing_enabled = _forcing_mode_enabled(forcing_mode)

    init_out = run_model_scan(
        M=int(M),
        dt=float(dt_seconds),
        Phibar=float(params.Phibar),
        omega=float(params.omega_rad_s),
        a=float(params.a_m),
        test=None,
        g=float(params.g_m_s2),
        forcflag=forcing_enabled,
        taurad=float(params.taurad_s),
        taudrag=float(params.taudrag_s),
        DPhieq=_forcing_mode_dphieq(forcing_mode, float(params.DPhieq)),
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
        **_initial_condition_kwargs(
            M=int(M),
            Phibar=float(params.Phibar),
            omega_rad_s=float(params.omega_rad_s),
            dt_seconds=float(dt_seconds),
            n_steps_total=int(n_steps_total),
            initial_condition_mode=initial_condition_mode,
            trajectory_seed=int(trajectory_seed),
            max_storms=int(max_storms),
            initial_phi_noise_temperature_k=float(initial_phi_noise_temperature_k),
            r_specific_j_per_kg_k=float(r_specific_j_per_kg_k),
            storm_radius_degrees=float(storm_radius_degrees),
            storm_nondim_lifetime=float(storm_nondim_lifetime),
            storm_nondim_interval=float(storm_nondim_interval),
            storm_strength_fraction=float(storm_strength_fraction),
        ),
    )
    current_state_full = init_out["last_state"]
    return (
        init_out["static"],
        current_state_full,
        jnp.asarray(current_state_full.U_curr),
        jnp.asarray(current_state_full.V_curr),
    )


def _initialize_trajectory_state_from_vector(
    param_vector: Any,
    *,
    M: int,
    dt_seconds: float,
    n_steps_total: int,
    starttime_index: int,
    k6: float,
    k6phi: float | None,
    forcing_mode: str,
    initial_condition_mode: str,
    trajectory_seed: Any,
    max_storms: int,
    initial_phi_noise_temperature_k: float,
    r_specific_j_per_kg_k: float,
    storm_radius_degrees: float,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
    storm_strength_fraction: float,
) -> tuple[Any, Any, Any, Any]:
    """Build static operators and initial state from a conditioning vector."""
    import jax.numpy as jnp
    from my_swamp.model import run_model_scan

    (
        a_m,
        omega_rad_s,
        Phibar,
        DPhieq,
        taurad_s,
        taudrag_s,
        g_m_s2,
    ) = param_vector
    forcing_enabled = _forcing_mode_enabled(forcing_mode)

    init_out = run_model_scan(
        M=int(M),
        dt=float(dt_seconds),
        Phibar=Phibar,
        omega=omega_rad_s,
        a=a_m,
        test=None,
        g=g_m_s2,
        forcflag=forcing_enabled,
        taurad=taurad_s,
        taudrag=taudrag_s,
        DPhieq=_forcing_mode_dphieq(forcing_mode, DPhieq),
        diffflag=True,
        modalflag=True,
        expflag=False,
        K6=float(k6),
        K6Phi=k6phi,
        diagnostics=False,
        return_history=False,
        starttime=int(starttime_index),
        tmax=int(starttime_index),
        jit_scan=True,
        donate_state=True,
        **_initial_condition_kwargs(
            M=int(M),
            Phibar=Phibar,
            omega_rad_s=omega_rad_s,
            dt_seconds=float(dt_seconds),
            n_steps_total=int(n_steps_total),
            initial_condition_mode=initial_condition_mode,
            trajectory_seed=trajectory_seed,
            max_storms=int(max_storms),
            initial_phi_noise_temperature_k=float(initial_phi_noise_temperature_k),
            r_specific_j_per_kg_k=float(r_specific_j_per_kg_k),
            storm_radius_degrees=float(storm_radius_degrees),
            storm_nondim_lifetime=float(storm_nondim_lifetime),
            storm_nondim_interval=float(storm_nondim_interval),
            storm_strength_fraction=float(storm_strength_fraction),
        ),
    )
    current_state_full = init_out["last_state"]
    return (
        init_out["static"],
        current_state_full,
        jnp.asarray(current_state_full.U_curr),
        jnp.asarray(current_state_full.V_curr),
    )


@lru_cache(maxsize=32)
def _get_reduced_carry_chunk_runner(
    *,
    starttime_index: int,
    total_time_seconds: float,
    convective_forcing_mode: str,
    max_storms: int,
    storm_padding_count: int,
    storm_radius_degrees: float,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
    storm_strength_fraction: float,
) -> Any:
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
        trajectory_seed: Any,
    ) -> tuple[Any, Any]:
        """Advance one chunk and collect visible states after each step."""
        import jax.numpy as jnp  # noqa: F811

        def _step(carry: Any, t: Any) -> tuple[Any, jnp.ndarray]:
            """Advance a single MY_SWAMP step inside the chunk scan."""
            new_state, _ = _step_once(carry, t, static, flags, None, Uic, Vic)
            new_state = _apply_convective_phi_update(
                state=new_state,
                static=static,
                t=t,
                starttime_index=int(starttime_index),
                total_time_seconds=float(total_time_seconds),
                trajectory_seed=trajectory_seed,
                convective_forcing_mode=convective_forcing_mode,
                max_storms=int(max_storms),
                storm_padding_count=int(storm_padding_count),
                storm_radius_degrees=float(storm_radius_degrees),
                storm_nondim_lifetime=float(storm_nondim_lifetime),
                storm_nondim_interval=float(storm_nondim_interval),
                storm_strength_fraction=float(storm_strength_fraction),
            )
            return new_state, _stack_reduced_carry_state_jax(new_state)

        return jax.lax.scan(_step, state0, t_seq)

    return jax.jit(_scan_chunk, donate_argnums=(2,))


@lru_cache(maxsize=8)
def _get_batched_trajectory_initializer(
    *,
    M: int,
    dt_seconds: float,
    n_steps_total: int,
    starttime_index: int,
    k6: float,
    k6phi: float | None,
    forcing_mode: str,
    initial_condition_mode: str,
    max_storms: int,
    initial_phi_noise_temperature_k: float,
    r_specific_j_per_kg_k: float,
    storm_radius_degrees: float,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
    storm_strength_fraction: float,
) -> Any:
    """Return a cached batched initializer for trajectory extraction."""
    import jax

    return jax.vmap(
        lambda param_vector, trajectory_seed: _initialize_trajectory_state_from_vector(
            param_vector,
            M=M,
            dt_seconds=dt_seconds,
            n_steps_total=n_steps_total,
            starttime_index=starttime_index,
            k6=k6,
            k6phi=k6phi,
            forcing_mode=forcing_mode,
            initial_condition_mode=initial_condition_mode,
            trajectory_seed=trajectory_seed,
            max_storms=max_storms,
            initial_phi_noise_temperature_k=initial_phi_noise_temperature_k,
            r_specific_j_per_kg_k=r_specific_j_per_kg_k,
            storm_radius_degrees=storm_radius_degrees,
            storm_nondim_lifetime=storm_nondim_lifetime,
            storm_nondim_interval=storm_nondim_interval,
            storm_strength_fraction=storm_strength_fraction,
        )
    )


@lru_cache(maxsize=8)
def _get_batched_checkpoint_runner(
    *,
    n_steps_total: int,
    dt_seconds: float,
    starttime_index: int,
    n_checkpoints: int,
    forcing_mode: str,
    convective_forcing_mode: str,
    max_storms: int,
    storm_padding_count: int,
    storm_radius_degrees: float,
    storm_nondim_lifetime: float,
    storm_nondim_interval: float,
    storm_strength_fraction: float,
) -> Any:
    """Return a cached batched rollout runner for uniform checkpoint sequences."""
    import jax
    import jax.numpy as jnp
    from my_swamp.model import _step_once_state_only

    flags = _build_run_flags(diagnostics=False, forcing_mode=forcing_mode)
    rel_steps = jnp.arange(1, int(n_steps_total) + 1, dtype=jnp.int32)
    current_field_indices = jnp.asarray(CURRENT_FIELD_INDICES, dtype=jnp.int32)

    def _step_one_sample(
        state_i: Any,
        checkpoint_buffer_i: Any,
        static_i: Any,
        Uic_i: Any,
        Vic_i: Any,
        checkpoint_steps_i: Any,
        max_checkpoint_step_i: Any,
        rel_step: Any,
        trajectory_seed_i: Any,
    ) -> tuple[Any, Any]:
        """Advance one sample by one step and materialize requested checkpoints."""

        def _do_step(_: None) -> tuple[Any, Any]:
            abs_t = jnp.asarray(int(starttime_index), dtype=jnp.int32) + rel_step - 1
            new_state = _step_once_state_only(
                state_i,
                abs_t,
                static_i,
                flags,
                None,
                Uic_i,
                Vic_i,
            )
            new_state = _apply_convective_phi_update(
                state=new_state,
                static=static_i,
                t=abs_t,
                starttime_index=int(starttime_index),
                total_time_seconds=float(n_steps_total) * float(dt_seconds),
                trajectory_seed=trajectory_seed_i,
                convective_forcing_mode=convective_forcing_mode,
                max_storms=int(max_storms),
                storm_padding_count=int(storm_padding_count),
                storm_radius_degrees=float(storm_radius_degrees),
                storm_nondim_lifetime=float(storm_nondim_lifetime),
                storm_nondim_interval=float(storm_nondim_interval),
                storm_strength_fraction=float(storm_strength_fraction),
            )
            reduced = _stack_reduced_carry_state_jax(new_state)
            current_fields = jnp.take(reduced, current_field_indices, axis=0)
            checkpoint_match = checkpoint_steps_i == rel_step
            checkpoint_buffer_next = jnp.where(
                checkpoint_match[:, None, None, None],
                current_fields[None, ...],
                checkpoint_buffer_i,
            )
            return new_state, checkpoint_buffer_next

        return jax.lax.cond(
            rel_step <= max_checkpoint_step_i,
            _do_step,
            lambda _: (state_i, checkpoint_buffer_i),
            operand=None,
        )

    def _run(
        static_batch: Any,
        state_batch: Any,
        Uic_batch: Any,
        Vic_batch: Any,
        checkpoint_steps_batch: Any,
        trajectory_seed_batch: Any,
    ) -> Any:
        """Advance a batch of trajectories and materialize checkpoint states."""
        reduced0 = jax.vmap(_stack_reduced_carry_state_jax)(state_batch)
        current_fields0 = jnp.take(reduced0, current_field_indices, axis=1)
        batch_size = int(current_fields0.shape[0])
        nlat = int(current_fields0.shape[-2])
        nlon = int(current_fields0.shape[-1])
        max_checkpoint_steps = jnp.max(checkpoint_steps_batch, axis=1)

        checkpoint_buffer = jnp.zeros(
            (batch_size, int(n_checkpoints), len(CURRENT_FIELD_INDICES), nlat, nlon),
            dtype=current_fields0.dtype,
        )
        checkpoint_buffer = jnp.where(
            checkpoint_steps_batch[:, :, None, None, None] == 0,
            current_fields0[:, None, ...],
            checkpoint_buffer,
        )

        def _scan_step(
            carry: tuple[Any, Any],
            rel_step: Any,
        ) -> tuple[tuple[Any, Any], None]:
            state_curr, checkpoint_curr = carry
            state_next, checkpoint_next = jax.vmap(
                _step_one_sample,
                in_axes=(0, 0, 0, 0, 0, 0, 0, None, 0),
            )(
                state_curr,
                checkpoint_curr,
                static_batch,
                Uic_batch,
                Vic_batch,
                checkpoint_steps_batch,
                max_checkpoint_steps,
                rel_step,
                trajectory_seed_batch,
            )
            return (state_next, checkpoint_next), None

        (_, checkpoint_buffer), _ = jax.lax.scan(
            _scan_step,
            (state_batch, checkpoint_buffer),
            rel_steps,
        )
        return checkpoint_buffer

    return jax.jit(_run)


def run_trajectory_checkpoints_batched(
    params_batch: np.ndarray,
    *,
    M: int,
    dt_seconds: float,
    time_days: float,
    starttime_index: int,
    checkpoint_steps_batch: np.ndarray,
    k6: float,
    k6phi: float | None,
    forcing_mode: str = "forced",
    initial_condition_mode: str = "legacy",
    convective_forcing_mode: str = "none",
    trajectory_seeds: np.ndarray | None = None,
    initial_phi_noise_temperature_k: float = 0.0,
    r_specific_j_per_kg_k: float = 3900.0,
    storm_radius_degrees: float = 2.0,
    storm_nondim_lifetime: float = 20.0,
    storm_nondim_interval: float = 20.0,
    storm_strength_fraction: float = 0.1,
) -> np.ndarray:
    """Extract many checkpoint sequences in parallel using one vectorized JAX rollout."""
    import jax
    import jax.numpy as jnp

    params_batch = np.asarray(params_batch, dtype=np.float64)
    checkpoint_steps_batch = np.asarray(checkpoint_steps_batch, dtype=np.int64)
    if trajectory_seeds is None:
        trajectory_seeds = np.zeros((params_batch.shape[0],), dtype=np.int64)
    trajectory_seeds = np.asarray(trajectory_seeds, dtype=np.int64)

    if params_batch.ndim != 2 or params_batch.shape[1] != len(CONDITIONING_PARAM_NAMES):
        raise ValueError(
            "params_batch must have shape "
            f"[B,{len(CONDITIONING_PARAM_NAMES)}], got {params_batch.shape}"
        )
    if checkpoint_steps_batch.ndim != 2:
        raise ValueError("checkpoint_steps_batch must be a rank-2 array")
    if checkpoint_steps_batch.shape[0] != params_batch.shape[0]:
        raise ValueError("Batch dimension of checkpoint_steps_batch must align with params_batch")
    if trajectory_seeds.shape != (params_batch.shape[0],):
        raise ValueError(
            "trajectory_seeds must have shape "
            f"[{params_batch.shape[0]}], got {trajectory_seeds.shape}"
        )
    if checkpoint_steps_batch.shape[1] < 1:
        raise ValueError("At least one checkpoint is required per batch element")
    if np.any(checkpoint_steps_batch < 0):
        raise ValueError("checkpoint_steps_batch must be >= 0")
    if time_days <= 0:
        raise ValueError("time_days must be > 0")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0")

    n_steps_total = _total_rollout_steps(time_days=time_days, dt_seconds=dt_seconds)
    if int(np.max(checkpoint_steps_batch)) > int(n_steps_total):
        raise ValueError(
            "Requested checkpoint exceeds the simulated horizon: "
            f"max_checkpoint_step={int(np.max(checkpoint_steps_batch))}, "
            f"available={int(n_steps_total)}"
        )

    param_batch_jax = jnp.asarray(params_batch, dtype=jnp.float64)
    checkpoint_steps_jax = jnp.asarray(checkpoint_steps_batch, dtype=jnp.int32)
    trajectory_seeds_jax = jnp.asarray(trajectory_seeds, dtype=jnp.int64)
    need_storm_table = (
        _convective_forcing_mode_enabled(str(convective_forcing_mode))
        or float(initial_phi_noise_temperature_k) > 0.0
    )
    max_storms = 0
    if need_storm_table:
        max_storms = _max_storm_count_for_omegas(
            n_steps_total=int(n_steps_total),
            dt_seconds=float(dt_seconds),
            omega_rad_s_values=params_batch[:, 1],
            storm_nondim_lifetime=float(storm_nondim_lifetime),
            storm_nondim_interval=float(storm_nondim_interval),
        )
    storm_padding_count = _storm_padding_count(
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
    )
    initializer = _get_batched_trajectory_initializer(
        M=int(M),
        dt_seconds=float(dt_seconds),
        n_steps_total=int(n_steps_total),
        starttime_index=int(starttime_index),
        k6=float(k6),
        k6phi=k6phi,
        forcing_mode=str(forcing_mode),
        initial_condition_mode=str(initial_condition_mode),
        max_storms=int(max_storms),
        initial_phi_noise_temperature_k=float(initial_phi_noise_temperature_k),
        r_specific_j_per_kg_k=float(r_specific_j_per_kg_k),
        storm_radius_degrees=float(storm_radius_degrees),
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
        storm_strength_fraction=float(storm_strength_fraction),
    )
    static_batch, state_batch, Uic_batch, Vic_batch = initializer(
        param_batch_jax,
        trajectory_seeds_jax,
    )
    runner = _get_batched_checkpoint_runner(
        n_steps_total=int(n_steps_total),
        dt_seconds=float(dt_seconds),
        starttime_index=int(starttime_index),
        n_checkpoints=int(checkpoint_steps_batch.shape[1]),
        forcing_mode=str(forcing_mode),
        convective_forcing_mode=str(convective_forcing_mode),
        max_storms=int(max_storms),
        storm_padding_count=int(storm_padding_count),
        storm_radius_degrees=float(storm_radius_degrees),
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
        storm_strength_fraction=float(storm_strength_fraction),
    )
    checkpoint_states = runner(
        static_batch,
        state_batch,
        Uic_batch,
        Vic_batch,
        checkpoint_steps_jax,
        trajectory_seeds_jax,
    )
    return np.asarray(jax.device_get(checkpoint_states), dtype=np.float64)


def run_trajectory_checkpoints(
    params: Extended9Params,
    *,
    M: int,
    dt_seconds: float,
    time_days: float,
    starttime_index: int,
    checkpoint_steps: np.ndarray,
    forcing_mode: str = "forced",
    initial_condition_mode: str = "legacy",
    convective_forcing_mode: str = "none",
    trajectory_seed: int = 0,
    initial_phi_noise_temperature_k: float = 0.0,
    r_specific_j_per_kg_k: float = 3900.0,
    storm_radius_degrees: float = 2.0,
    storm_nondim_lifetime: float = 20.0,
    storm_nondim_interval: float = 20.0,
    storm_strength_fraction: float = 0.1,
) -> np.ndarray:
    """Extract visible-state checkpoints from one MY_SWAMP rollout."""
    import jax.numpy as jnp

    checkpoint_steps = np.asarray(checkpoint_steps, dtype=np.int64)

    if checkpoint_steps.ndim != 1:
        raise ValueError("checkpoint_steps must be a 1-D array")
    if checkpoint_steps.shape[0] < 1:
        raise ValueError("At least one checkpoint is required")
    if time_days <= 0:
        raise ValueError("time_days must be > 0")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0")
    if np.any(checkpoint_steps < 0):
        raise ValueError("checkpoint_steps must be >= 0")

    n_steps_total = _total_rollout_steps(time_days=time_days, dt_seconds=dt_seconds)
    if int(np.max(checkpoint_steps)) > n_steps_total:
        raise ValueError(
            "Requested checkpoint exceeds the simulated horizon: "
            f"max_checkpoint_step={int(np.max(checkpoint_steps))}, "
            f"available={n_steps_total}"
        )

    required_steps = np.unique(
        np.concatenate([np.asarray([0], dtype=np.int64), checkpoint_steps], axis=0)
    )
    need_storm_table = (
        _convective_forcing_mode_enabled(str(convective_forcing_mode))
        or float(initial_phi_noise_temperature_k) > 0.0
    )
    max_storms = 0
    if need_storm_table:
        max_storms = _max_storm_count_for_omegas(
            n_steps_total=int(n_steps_total),
            dt_seconds=float(dt_seconds),
            omega_rad_s_values=np.asarray([float(params.omega_rad_s)], dtype=np.float64),
            storm_nondim_lifetime=float(storm_nondim_lifetime),
            storm_nondim_interval=float(storm_nondim_interval),
        )
    storm_padding_count = _storm_padding_count(
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
    )

    static, current_state_full, Uic, Vic = _initialize_trajectory_state(
        params,
        M=int(M),
        dt_seconds=float(dt_seconds),
        n_steps_total=int(n_steps_total),
        starttime_index=int(starttime_index),
        forcing_mode=str(forcing_mode),
        initial_condition_mode=str(initial_condition_mode),
        trajectory_seed=int(trajectory_seed),
        max_storms=int(max_storms),
        initial_phi_noise_temperature_k=float(initial_phi_noise_temperature_k),
        r_specific_j_per_kg_k=float(r_specific_j_per_kg_k),
        storm_radius_degrees=float(storm_radius_degrees),
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
        storm_strength_fraction=float(storm_strength_fraction),
    )
    flags = _build_run_flags(diagnostics=False, forcing_mode=str(forcing_mode))
    states_by_step: Dict[int, np.ndarray] = {
        0: _snapshot_from_last_state(current_state_full).as_array().astype(
            np.float64, copy=False
        )
    }
    required_list = [int(value) for value in required_steps.tolist()]
    req_ptr = 1
    step_cursor = 0
    chunk_runner = _get_reduced_carry_chunk_runner(
        starttime_index=int(starttime_index),
        total_time_seconds=float(n_steps_total) * float(dt_seconds),
        convective_forcing_mode=str(convective_forcing_mode),
        max_storms=int(max_storms),
        storm_padding_count=int(storm_padding_count),
        storm_radius_degrees=float(storm_radius_degrees),
        storm_nondim_lifetime=float(storm_nondim_lifetime),
        storm_nondim_interval=float(storm_nondim_interval),
        storm_strength_fraction=float(storm_strength_fraction),
    )

    while req_ptr < len(required_list):
        if step_cursor >= n_steps_total:
            raise RuntimeError(
                "Failed to collect all requested trajectory checkpoints before horizon end"
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
            jnp.asarray(int(trajectory_seed), dtype=jnp.int64),
        )
        chunk_history_np = np.asarray(chunk_history, dtype=np.float64)
        chunk_end = step_cursor + chunk_len

        while req_ptr < len(required_list) and required_list[req_ptr] <= chunk_end:
            req_step = required_list[req_ptr]
            rel = req_step - step_cursor
            if rel < 1:
                raise RuntimeError(
                    f"Invalid relative checkpoint offset {rel} for req_step={req_step}"
                )
            states_by_step[req_step] = chunk_history_np[rel - 1].astype(
                np.float64, copy=False
            )
            req_ptr += 1

        step_cursor = chunk_end

    current_field_indices = list(CURRENT_FIELD_INDICES)
    return np.stack(
        [
            np.take(states_by_step[int(step)], current_field_indices, axis=0)
            for step in checkpoint_steps.tolist()
        ],
        axis=0,
    ).astype(np.float64)


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
    forcing_mode: str,
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
        DPhieq=_forcing_mode_dphieq(forcing_mode, float(DPhieq)),
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
    forcing_mode: str = "forced",
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagnose physical-space ``U,V`` from physical-space ``eta,delta``.

    Args:
        eta:
            Physical-space vorticity-like field with shape ``[H, W]``.
        delta:
            Physical-space divergence field with shape ``[H, W]``.

    Returns:
        Two ``[H, W]`` arrays containing the reconstructed ``U`` and ``V``
        fields.
    """
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
        forcing_mode=str(forcing_mode),
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
    forcing_mode: str = "forced",
) -> np.ndarray:
    """Reconstruct a full physical 5-field state from prognostic ``Phi,eta,delta``.

    Args:
        prognostics:
            Physical prognostic tensor with shape ``[3, H, W]`` ordered as
            ``(Phi, eta, delta)``.

    Returns:
        Full visible state with shape ``[5, H, W]`` ordered as
        ``(Phi, U, V, eta, delta)``.
    """
    if prognostics.shape[0] != len(PROGNOSTIC_STATE_FIELDS):
        raise ValueError(
            "prognostics must have "
            f"{len(PROGNOSTIC_STATE_FIELDS)} channels, "
            f"got {prognostics.shape[0]}"
        )
    phi = np.asarray(prognostics[0], dtype=np.float64)
    eta = np.asarray(prognostics[1], dtype=np.float64)
    delta = np.asarray(prognostics[2], dtype=np.float64)
    u_field, v_field = diagnose_winds(
        eta,
        delta,
        params=params,
        M=M,
        dt_seconds=dt_seconds,
        forcing_mode=forcing_mode,
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
