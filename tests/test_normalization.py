from __future__ import annotations

import numpy as np

from src.config import NormalizationConfig, ParamNormConfig
from src.normalization import denormalize_states, fit_normalization, normalize_params, normalize_states


def test_normalization_roundtrip_with_transforms() -> None:
    rng = np.random.default_rng(2)

    # build positive Phi for log10 and signed fields for signed_log1p
    states = np.zeros((10, 5, 4, 8), dtype=np.float32)
    states[:, 0] = np.abs(rng.normal(100.0, 10.0, size=(10, 4, 8))).astype(np.float32)  # Phi
    states[:, 1] = rng.normal(0.0, 1.0, size=(10, 4, 8)).astype(np.float32)             # U
    states[:, 2] = rng.normal(0.0, 1.0, size=(10, 4, 8)).astype(np.float32)             # V
    states[:, 3] = rng.normal(0.0, 5.0, size=(10, 4, 8)).astype(np.float32)             # eta
    states[:, 4] = rng.normal(0.0, 5.0, size=(10, 4, 8)).astype(np.float32)             # delta

    params = rng.normal(size=(10, 9)).astype(np.float64)

    cfg = NormalizationConfig(
        field_transforms={"Phi": "log10", "U": "none", "V": "none", "eta": "signed_log1p", "delta": "signed_log1p"},
        zscore_eps=1e-6,
        log10_eps=1e-20,
        signed_log1p_scale=2.0,
    )

    stats = fit_normalization(
        train_states_nchw=states,
        train_params_np=params,
        field_names=["Phi", "U", "V", "eta", "delta"],
        param_names=[f"p{i}" for i in range(9)],
        state_cfg=cfg,
        param_cfg=ParamNormConfig(mode="zscore", eps=1e-6),
    )

    st_norm = normalize_states(states, stats)
    st_back = denormalize_states(st_norm, stats)

    np.testing.assert_allclose(st_back, states, rtol=1e-4, atol=1e-4)

    p_norm = normalize_params(params, stats)
    assert p_norm.shape == (10, 9)
