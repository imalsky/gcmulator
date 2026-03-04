from __future__ import annotations

import numpy as np

from src.geometry import apply_geometry_state


def test_geometry_flip_and_roll() -> None:
    # [C,H,W]
    arr = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
    out, info = apply_geometry_state(
        arr,
        flip_latitude_to_north_south=True,
        roll_longitude_to_0_2pi=True,
    )

    # expected: latitude reversed then longitude rolled by -W/2 = -3
    exp = arr[:, ::-1, :]
    exp = np.roll(exp, shift=-3, axis=-1)
    np.testing.assert_allclose(out, exp)
    assert info["lat_order"] == "north_to_south"
    assert info["lon_origin"] == "0_to_2pi"
    assert info["lon_shift"] == -3
