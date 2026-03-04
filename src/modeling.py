from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .constants import RANDOM_BASIS_AMP_MAX, RANDOM_BASIS_AMP_MIN, TWO_PI

def ensure_torch_harmonics_importable(config_dir: Path) -> None:
    try:
        import torch_harmonics  # noqa: F401
        return
    except Exception:
        pass

    candidates = [
        config_dir / "torch-harmonics-main",
        config_dir / "torch-harmonics",
        config_dir.parent / "torch-harmonics-main",
        config_dir.parent / "torch-harmonics",
    ]
    for c in candidates:
        if (c / "torch_harmonics").is_dir():
            os.sys.path.insert(0, str(c))
            try:
                import torch_harmonics  # noqa: F401
                return
            except Exception:
                os.sys.path.pop(0)

    raise RuntimeError("Could not import torch_harmonics. Install it or keep torch-harmonics-main nearby.")


def import_sfno() -> type[nn.Module]:
    from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator  # type: ignore

    return SphericalFourierNeuralOperator


def choose_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device='cuda' but CUDA is unavailable")
        return torch.device("cuda")
    if device_cfg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device='mps' but MPS is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, amp_mode: str):
    if amp_mode == "bf16" and device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if amp_mode == "fp16" and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


class SphereLoss(nn.Module):
    """Quadrature-weighted squared L2 over sphere, mean over batch and channels."""

    def __init__(self, nlat: int, nlon: int, grid: str) -> None:
        super().__init__()
        from torch_harmonics.examples.losses import get_quadrature_weights  # type: ignore

        q = get_quadrature_weights(nlat=nlat, nlon=nlon, grid=grid, tile=False, normalized=True)
        self.register_buffer("quad", q.to(torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch pred={tuple(pred.shape)} target={tuple(target.shape)}")
        q = self.quad
        if q.ndim == 2:
            q = q[:, 0]
        q = q[None, None, :, None].to(device=pred.device, dtype=pred.dtype)
        loss = ((pred - target) ** 2) * q
        loss = loss.sum(dim=(-2, -1)).mean(dim=(-1, -2))
        return loss.mean()


class ConditionalResidualWrapper(nn.Module):
    def __init__(self, base: nn.Module, state_chans: int, residual_prediction: bool) -> None:
        super().__init__()
        self.base = base
        self.state_chans = int(state_chans)
        self.residual_prediction = bool(residual_prediction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.residual_prediction:
            y = y + x[:, : self.state_chans]
        return y


def build_coord_channels_legendre_gauss(
    nlat: int,
    nlon: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mu, _ = np.polynomial.legendre.leggauss(int(nlat))
    mu = mu[::-1].copy()
    lat = np.arcsin(mu).astype(np.float32)
    lon = (np.arange(int(nlon), dtype=np.float32) * (TWO_PI / float(nlon))).astype(np.float32)

    lat2d = lat[:, None]
    lon2d = lon[None, :]

    sin_lat = np.repeat(np.sin(lat2d), int(nlon), axis=1)
    cos_lat = np.repeat(np.cos(lat2d), int(nlon), axis=1)
    sin_lon = np.repeat(np.sin(lon2d), int(nlat), axis=0)
    cos_lon = np.repeat(np.cos(lon2d), int(nlat), axis=0)

    out = np.stack([sin_lat, cos_lat, sin_lon, cos_lon], axis=0).astype(np.float32)
    return torch.from_numpy(out).to(device=device, dtype=dtype)


def _make_smooth_random_basis(
    nlat: int,
    nlon: int,
    *,
    seed: int,
    count: int,
    max_k: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    mu, _ = np.polynomial.legendre.leggauss(int(nlat))
    mu = mu[::-1].copy()
    lat = np.arcsin(mu).astype(np.float32)
    lon = (np.arange(int(nlon), dtype=np.float32) * (TWO_PI / float(nlon))).astype(np.float32)
    lat2d = lat[:, None]
    lon2d = lon[None, :]

    maps: List[np.ndarray] = []
    for _ in range(int(count)):
        k = int(rng.integers(1, max_k + 1))
        m = int(rng.integers(1, max_k + 1))
        phase = float(rng.uniform(0.0, TWO_PI))
        amp = float(rng.uniform(RANDOM_BASIS_AMP_MIN, RANDOM_BASIS_AMP_MAX))
        arr = (amp * np.cos(m * lat2d) * np.sin(k * lon2d + phase)).astype(np.float32)
        maps.append(arr)
    return np.stack(maps, axis=0)


class ParamICBasisMLP(nn.Module):
    def __init__(
        self,
        *,
        param_dim: int,
        state_chans: int,
        nlat: int,
        nlon: int,
        basis: Sequence[str],
        hidden_dim: int,
        num_layers: int,
        activation: str,
        rand_basis_seed: int,
        rand_basis_count: int,
        rand_basis_max_k: int,
        out_tanh: bool,
    ) -> None:
        super().__init__()
        self.param_dim = int(param_dim)
        self.state_chans = int(state_chans)
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.out_tanh = bool(out_tanh)

        basis_maps: List[np.ndarray] = []
        coords = build_coord_channels_legendre_gauss(nlat, nlon, dtype=torch.float32, device=torch.device("cpu")).numpy()
        sin_lat, cos_lat, sin_lon, cos_lon = coords

        for name in [str(x).lower() for x in basis]:
            if name == "const":
                basis_maps.append(np.ones((nlat, nlon), dtype=np.float32))
            elif name == "sin_lat":
                basis_maps.append(sin_lat)
            elif name == "cos_lat":
                basis_maps.append(cos_lat)
            elif name == "sin_lon":
                basis_maps.append(sin_lon)
            elif name == "cos_lon":
                basis_maps.append(cos_lon)

        if rand_basis_count > 0:
            rnd = _make_smooth_random_basis(
                nlat=nlat,
                nlon=nlon,
                seed=rand_basis_seed,
                count=rand_basis_count,
                max_k=rand_basis_max_k,
            )
            for i in range(rnd.shape[0]):
                basis_maps.append(rnd[i])

        if not basis_maps:
            raise ValueError("IC basis map set is empty")

        basis_arr = np.stack(basis_maps, axis=0).astype(np.float32)
        self.register_buffer("basis_maps", torch.from_numpy(basis_arr), persistent=False)
        self.nbasis = int(basis_arr.shape[0])

        act_map: Dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported IC activation: {activation}")

        layers: List[nn.Module] = []
        in_dim = int(param_dim)
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(act_map[activation])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, self.state_chans * self.nbasis))
        self.mlp = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if params.ndim != 2 or params.shape[-1] != self.param_dim:
            raise ValueError(f"params must be [B,{self.param_dim}], got {tuple(params.shape)}")

        bsz = int(params.shape[0])
        coeff = self.mlp(params).view(bsz, self.state_chans, self.nbasis)
        basis = self.basis_maps.to(device=params.device, dtype=params.dtype)
        state0 = torch.einsum("bcn,nhw->bchw", coeff, basis)
        if self.out_tanh:
            state0 = torch.tanh(state0)
        return state0


class ParamRolloutModel(nn.Module):
    def __init__(
        self,
        *,
        stepper: nn.Module,
        ic: ParamICBasisMLP,
        param_dim: int,
        state_chans: int,
        nlat: int,
        nlon: int,
        include_param_maps: bool,
        include_coord_channels: bool,
    ) -> None:
        super().__init__()
        self.stepper = stepper
        self.ic = ic
        self.param_dim = int(param_dim)
        self.state_chans = int(state_chans)
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.include_param_maps = bool(include_param_maps)
        self.include_coord_channels = bool(include_coord_channels)

        if self.include_coord_channels:
            cc = build_coord_channels_legendre_gauss(nlat, nlon, dtype=torch.float32, device=torch.device("cpu"))
            self.register_buffer("coord_channels", cc, persistent=False)
        else:
            self.register_buffer("coord_channels", torch.zeros((0, nlat, nlon), dtype=torch.float32), persistent=False)

    def _param_maps(self, params: torch.Tensor) -> torch.Tensor:
        bsz, p = params.shape
        return params[:, :, None, None].expand(bsz, p, self.nlat, self.nlon)

    def rollout(self, params: torch.Tensor, *, steps: int) -> torch.Tensor:
        if steps < 1:
            raise ValueError("rollout steps must be >= 1")
        if params.ndim != 2 or params.shape[-1] != self.param_dim:
            raise ValueError(f"params must be [B,{self.param_dim}], got {tuple(params.shape)}")

        state = self.ic(params)

        coord = self.coord_channels.to(device=params.device, dtype=params.dtype)
        if coord.numel():
            coord = coord.unsqueeze(0).expand(params.shape[0], -1, -1, -1)

        pmap = self._param_maps(params) if self.include_param_maps else None

        for _ in range(int(steps)):
            parts = [state]
            if pmap is not None:
                parts.append(pmap)
            if coord.numel():
                parts.append(coord)
            x = torch.cat(parts, dim=1)
            state = self.stepper(x)

        return state

    def forward(self, params: torch.Tensor, steps: int) -> torch.Tensor:
        return self.rollout(params, steps=int(steps))


def build_rollout_model(
    *,
    img_size: tuple[int, int],
    state_chans: int,
    param_dim: int,
    cfg_model,
) -> ParamRolloutModel:
    h, w = int(img_size[0]), int(img_size[1])

    coord_chans = 4 if cfg_model.include_coord_channels else 0
    in_chans = state_chans + (param_dim if cfg_model.include_param_maps else 0) + coord_chans
    out_chans = state_chans

    SphericalFNO = import_sfno()
    base = SphericalFNO(
        img_size=(h, w),
        grid=cfg_model.grid,
        grid_internal=cfg_model.grid_internal,
        scale_factor=int(cfg_model.scale_factor),
        in_chans=int(in_chans),
        out_chans=int(out_chans),
        embed_dim=int(cfg_model.embed_dim),
        num_layers=int(cfg_model.num_layers),
        activation_function=str(cfg_model.activation_function),
        encoder_layers=int(cfg_model.encoder_layers),
        use_mlp=bool(cfg_model.use_mlp),
        mlp_ratio=float(cfg_model.mlp_ratio),
        drop_rate=float(cfg_model.drop_rate),
        drop_path_rate=float(cfg_model.drop_path_rate),
        normalization_layer=str(cfg_model.normalization_layer),
        hard_thresholding_fraction=float(cfg_model.hard_thresholding_fraction),
        residual_prediction=False,
        pos_embed=str(cfg_model.pos_embed),
        bias=bool(cfg_model.bias),
    )

    stepper = ConditionalResidualWrapper(base, state_chans=state_chans, residual_prediction=bool(cfg_model.residual_prediction))

    ic = ParamICBasisMLP(
        param_dim=param_dim,
        state_chans=state_chans,
        nlat=h,
        nlon=w,
        basis=list(cfg_model.ic.basis),
        hidden_dim=int(cfg_model.ic.hidden_dim),
        num_layers=int(cfg_model.ic.num_layers),
        activation=str(cfg_model.ic.activation),
        rand_basis_seed=int(cfg_model.ic.rand_basis_seed),
        rand_basis_count=int(cfg_model.ic.rand_basis_count),
        rand_basis_max_k=int(cfg_model.ic.rand_basis_max_k),
        out_tanh=bool(cfg_model.ic.out_tanh),
    )

    return ParamRolloutModel(
        stepper=stepper,
        ic=ic,
        param_dim=param_dim,
        state_chans=state_chans,
        nlat=h,
        nlon=w,
        include_param_maps=bool(cfg_model.include_param_maps),
        include_coord_channels=bool(cfg_model.include_coord_channels),
    )
