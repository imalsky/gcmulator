"""Helpers for aligning latitude/longitude grid conventions.

The emulator stores physical states in a single canonical orientation. These
utilities keep the conversion logic in one place so data generation,
preprocessing, and evaluation all agree on what "north-to-south" and
"0-to-2pi" mean.
"""

from __future__ import annotations


def geometry_shift_for_nlon(nlon: int, roll_longitude_to_0_2pi: bool) -> int:
    """Return the longitude roll needed to move ``[-pi, pi)`` to ``[0, 2pi)``."""
    if not roll_longitude_to_0_2pi:
        return 0
    if nlon % 2 != 0:
        raise ValueError(f"roll_longitude_to_0_2pi requires even nlon, got {nlon}")
    return -(nlon // 2)
