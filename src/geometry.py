"""Grid-orientation helpers for latitude/longitude convention alignment."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def geometry_shift_for_nlon(nlon: int, roll_longitude_to_0_2pi: bool) -> int:
    """Return longitude roll offset needed to move origin from [-pi,pi) to [0,2pi)."""
    if not roll_longitude_to_0_2pi:
        return 0
    if nlon % 2 != 0:
        raise ValueError(f"roll_longitude_to_0_2pi requires even nlon, got {nlon}")
    return -(nlon // 2)


def apply_geometry_state(
    state_chw: np.ndarray,
    *,
    flip_latitude_to_north_south: bool,
    roll_longitude_to_0_2pi: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply geometry convention conversion on a single [C,H,W] state."""
    if state_chw.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got shape {state_chw.shape}")

    _, nlat, nlon = state_chw.shape
    lon_shift = geometry_shift_for_nlon(nlon, roll_longitude_to_0_2pi)

    out = state_chw
    if flip_latitude_to_north_south:
        out = out[:, ::-1, :]
    if lon_shift:
        out = np.roll(out, shift=int(lon_shift), axis=-1)

    info = {
        "lat_order": "north_to_south" if flip_latitude_to_north_south else "south_to_north",
        "lon_origin": "0_to_2pi" if roll_longitude_to_0_2pi else "minus_pi_to_pi",
        "lon_shift": int(lon_shift),
        "nlat": int(nlat),
        "nlon": int(nlon),
    }
    return out, info
