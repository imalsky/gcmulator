from __future__ import annotations

import numpy as np

from src.config import ParameterSpec
from src.sampling import sample_parameter_dict, to_extended9


def test_sampling_bounds_and_k6phi_mixture() -> None:
    specs = [
        ParameterSpec(name="a_m", dist="loguniform", min=3e7, max=2e8),
        ParameterSpec(name="omega_rad_s", dist="loguniform", min=1e-6, max=1e-4),
        ParameterSpec(name="Phibar", dist="loguniform", min=1e5, max=1e6),
        ParameterSpec(name="DPhieq", dist="loguniform", min=1e5, max=5e6),
        ParameterSpec(name="taurad_hours", dist="uniform", min=1.0, max=30.0),
        ParameterSpec(name="taudrag_hours", dist="uniform", min=1.0, max=30.0),
        ParameterSpec(name="g_m_s2", dist="loguniform", min=1.0, max=40.0),
        ParameterSpec(name="K6", dist="loguniform", min=1e31, max=1e35),
        ParameterSpec(name="K6Phi", dist="mixture_off_loguniform", p_off=0.5, off_value=0.0, on_min=1e30, on_max=1e34),
    ]

    rng = np.random.default_rng(123)
    vals = [sample_parameter_dict(rng, specs) for _ in range(200)]

    assert all(3e7 <= v["a_m"] <= 2e8 for v in vals)
    assert all(1e-6 <= v["omega_rad_s"] <= 1e-4 for v in vals)
    assert all(1.0 * 3600.0 <= v["taurad_s"] <= 30.0 * 3600.0 for v in vals)
    assert all(1.0 * 3600.0 <= v["taudrag_s"] <= 30.0 * 3600.0 for v in vals)

    # mixture should produce at least one off and one on sample with high probability for n=200
    off_count = sum(float(v["K6Phi"]) == 0.0 for v in vals)
    on_count = len(vals) - off_count
    assert off_count > 0
    assert on_count > 0

    ext = to_extended9(vals[0])
    assert ext.to_vector().shape == (9,)
