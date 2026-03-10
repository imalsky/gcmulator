"""Helpers for aligning latitude/longitude grid conventions.

The emulator stores physical states in a single canonical orientation. These
utilities keep the conversion logic in one place so data generation,
preprocessing, and evaluation all agree on what "north-to-south" and
"0-to-2pi" mean.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def geometry_shift_for_nlon(nlon: int, roll_longitude_to_0_2pi: bool) -> int:
    """Return the longitude roll needed to move ``[-pi, pi)`` to ``[0, 2pi)``."""
    if not roll_longitude_to_0_2pi:
        return 0
    if nlon % 2 != 0:
        raise ValueError(f"roll_longitude_to_0_2pi requires even nlon, got {nlon}")
    return -(nlon // 2)


def apply_geometry_state(
    state: np.ndarray,
    *,
    flip_latitude_to_north_south: bool,
    roll_longitude_to_0_2pi: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply the canonical geometry transform on an array ending in ``[..., H, W]``."""
    if state.ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions ending in [C,H,W], got {state.shape}")

    nlat = int(state.shape[-2])
    nlon = int(state.shape[-1])
    lon_shift = geometry_shift_for_nlon(nlon, roll_longitude_to_0_2pi)

    out = np.asarray(state)
    if flip_latitude_to_north_south:
        out = np.flip(out, axis=-2)
    if lon_shift:
        out = np.roll(out, shift=int(lon_shift), axis=-1)
    out = np.ascontiguousarray(out)

    info = {
        "lat_order": (
            "north_to_south" if flip_latitude_to_north_south else "south_to_north"
        ),
        "lon_origin": "0_to_2pi" if roll_longitude_to_0_2pi else "minus_pi_to_pi",
        "lon_shift": int(lon_shift),
        "nlat": int(nlat),
        "nlon": int(nlon),
    }
    return out, info
