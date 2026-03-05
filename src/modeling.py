"""Model-building utilities for conditional spherical rollout emulation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

# Basis construction constants.
TWO_PI = 2.0 * np.pi
RANDOM_BASIS_AMP_MIN = 0.25
RANDOM_BASIS_AMP_MAX = 1.0

def ensure_torch_harmonics_importable(config_dir: Path) -> None:
    """Ensure ``torch_harmonics`` is importable from install or nearby checkout."""
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

    attempted = ", ".join(str(c) for c in candidates)
    raise RuntimeError(
        "Could not import torch_harmonics. Install it (for example: "
        "pip install 'git+https://github.com/NVIDIA/torch-harmonics.git') "
        "or run through run.pbs/run.sh with TORCH_HARMONICS_PACKAGE_SPEC set. "
        f"Also searched local checkout candidates: {attempted}."
    )


def import_sfno() -> type[nn.Module]:
    """Import the SFNO model class with cross-version compatibility shim."""
    import torch_harmonics.examples.models.sfno as sfno_mod  # type: ignore
    # Compatibility shim for torch-harmonics versions where SFNO references
    # DropPath but does not import it in sfno.py.
    if not hasattr(sfno_mod, "DropPath"):
        from torch_harmonics.examples.models._layers import DropPath  # type: ignore

        sfno_mod.DropPath = DropPath
    SphericalFourierNeuralOperator = sfno_mod.SphericalFourierNeuralOperator

    return SphericalFourierNeuralOperator


def _count_pointwise_convs(seq: nn.Module) -> Optional[int]:
    """Count 1x1 Conv2d layers in an ``nn.Sequential`` encoder block."""
    if not isinstance(seq, nn.Sequential):
        return None
    return sum(1 for m in seq if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1))


def _build_pointwise_stack(
    *,
    in_chans: int,
    out_chans: int,
    num_layers: int,
    hidden_dim: int,
    activation_fn: type[nn.Module],
    final_bias: bool,
) -> nn.Sequential:
    """Build a pointwise Conv2d MLP stack used as SFNO encoder override."""
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    layers: List[nn.Module] = []
    current_dim = int(in_chans)
    for _ in range(int(num_layers) - 1):
        fc = nn.Conv2d(current_dim, int(hidden_dim), 1, bias=True)
        nn.init.normal_(fc.weight, mean=0.0, std=math.sqrt(2.0 / current_dim))
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        layers.append(fc)
        layers.append(activation_fn())
        current_dim = int(hidden_dim)

    fc = nn.Conv2d(current_dim, int(out_chans), 1, bias=bool(final_bias))
    nn.init.normal_(fc.weight, mean=0.0, std=math.sqrt(1.0 / current_dim))
    if fc.bias is not None:
        nn.init.constant_(fc.bias, 0.0)
    layers.append(fc)
    return nn.Sequential(*layers)


def _ensure_sfno_encoder_depth(
    *,
    base: nn.Module,
    in_chans: int,
    encoder_layers: int,
    mlp_ratio: float,
    bias: bool,
) -> None:
    """Patch SFNO encoder depth when library defaults ignore ``encoder_layers``."""
    desired_layers = max(1, int(encoder_layers))
    existing_layers = _count_pointwise_convs(getattr(base, "encoder", nn.Identity()))
    if existing_layers == desired_layers:
        return

    activation_fn = getattr(base, "activation_function", nn.GELU)
    if not isinstance(activation_fn, type) or not issubclass(activation_fn, nn.Module):
        activation_fn = nn.GELU

    embed_dim = int(getattr(base, "embed_dim"))
    hidden_dim = max(1, int(round(embed_dim * float(mlp_ratio))))
    base.encoder = _build_pointwise_stack(
        in_chans=int(in_chans),
        out_chans=embed_dim,
        num_layers=desired_layers,
        hidden_dim=hidden_dim,
        activation_fn=activation_fn,
        final_bias=bool(bias),
    )
    if hasattr(base, "encoder_layers"):
        setattr(base, "encoder_layers", desired_layers)


def choose_device(device_cfg: str) -> torch.device:
    """Resolve runtime torch device from config string."""
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device='cuda' but CUDA is unavailable")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_context(device: torch.device, amp_mode: str):
    """Return autocast context manager for configured AMP mode."""
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
        """Compute quadrature-weighted mean-squared error on the sphere."""
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch pred={tuple(pred.shape)} target={tuple(target.shape)}")
        q = self.quad
        if q.ndim == 2:
            q = q[:, 0]
        q = q[None, None, :, None].to(device=pred.device, dtype=pred.dtype)
        loss = ((pred - target) ** 2) * q
        return loss.sum(dim=(-2, -1)).mean()


class ConditionalResidualWrapper(nn.Module):
    """Wrap one-step model and optionally apply residual state update."""

    def __init__(
        self,
        base: nn.Module,
        state_chans: int,
        residual_prediction: bool,
        residual_init_scale: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.state_chans = int(state_chans)
        self.residual_prediction = bool(residual_prediction)
        scale_init = torch.tensor(float(residual_init_scale), dtype=torch.float32)
        if self.residual_prediction:
            self.residual_scale = nn.Parameter(scale_init)
        else:
            self.register_buffer("residual_scale", scale_init, persistent=False)

    def forward(self, x: torch.Tensor, *, step_scale: float = 1.0) -> torch.Tensor:
        """Run step model and add scaled residual to state channels when enabled."""
        y = self.base(x)
        if self.residual_prediction:
            scale = self.residual_scale.to(device=y.device, dtype=y.dtype) * float(step_scale)
            y = x[:, : self.state_chans] + scale * y
        return y


def build_coord_channels_legendre_gauss(
    nlat: int,
    nlon: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    lat_order: str = "north_to_south",
    lon_origin: str = "0_to_2pi",
) -> torch.Tensor:
    """Build fixed sin/cos latitude-longitude channels on Legendre-Gauss grid."""
    mu, _ = np.polynomial.legendre.leggauss(int(nlat))
    if lat_order == "north_to_south":
        mu = mu[::-1].copy()
    elif lat_order == "south_to_north":
        mu = mu.copy()
    else:
        raise ValueError(f"Unsupported lat_order: {lat_order}")
    lat = np.arcsin(mu).astype(np.float32)
    if lon_origin == "0_to_2pi":
        lon = (np.arange(int(nlon), dtype=np.float32) * (TWO_PI / float(nlon))).astype(np.float32)
    elif lon_origin == "minus_pi_to_pi":
        lon = np.linspace(-np.pi, np.pi, num=int(nlon), endpoint=False, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported lon_origin: {lon_origin}")

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
    lat_order: str = "north_to_south",
    lon_origin: str = "0_to_2pi",
) -> np.ndarray:
    """Generate smooth random basis maps for IC parameterization."""
    rng = np.random.default_rng(int(seed))
    mu, _ = np.polynomial.legendre.leggauss(int(nlat))
    if lat_order == "north_to_south":
        mu = mu[::-1].copy()
    elif lat_order == "south_to_north":
        mu = mu.copy()
    else:
        raise ValueError(f"Unsupported lat_order: {lat_order}")
    lat = np.arcsin(mu).astype(np.float32)
    if lon_origin == "0_to_2pi":
        lon = (np.arange(int(nlon), dtype=np.float32) * (TWO_PI / float(nlon))).astype(np.float32)
    elif lon_origin == "minus_pi_to_pi":
        lon = np.linspace(-np.pi, np.pi, num=int(nlon), endpoint=False, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported lon_origin: {lon_origin}")
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
    """Map normalized parameters to an initial state using spatial basis maps."""

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
        lat_order: str = "north_to_south",
        lon_origin: str = "0_to_2pi",
    ) -> None:
        super().__init__()
        self.param_dim = int(param_dim)
        self.state_chans = int(state_chans)
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.out_tanh = bool(out_tanh)

        basis_maps: List[np.ndarray] = []
        coords = build_coord_channels_legendre_gauss(
            nlat,
            nlon,
            dtype=torch.float32,
            device=torch.device("cpu"),
            lat_order=str(lat_order),
            lon_origin=str(lon_origin),
        ).numpy()
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
                lat_order=str(lat_order),
                lon_origin=str(lon_origin),
            )
            for i in range(rnd.shape[0]):
                basis_maps.append(rnd[i])

        if not basis_maps:
            raise ValueError("IC basis map set is empty")

        basis_arr = np.stack(basis_maps, axis=0).astype(np.float32)
        self.register_buffer("basis_maps", torch.from_numpy(basis_arr), persistent=False)
        self.nbasis = int(basis_arr.shape[0])

        act_map: Dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported IC activation: {activation}")

        layers: List[nn.Module] = []
        in_dim = int(param_dim)
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(act_map[activation]())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, self.state_chans * self.nbasis))
        self.mlp = nn.Sequential(*layers)
        # Start from a neutral IC map so early rollouts are stable; the network learns
        # non-zero IC structure during training if needed.
        final = self.mlp[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Predict initial state tensor ``[B,C,H,W]`` from parameter vectors."""
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
    """Legacy parameter-only autoregressive rollout model."""

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
        lat_order: str = "north_to_south",
        lon_origin: str = "0_to_2pi",
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
            cc = build_coord_channels_legendre_gauss(
                nlat,
                nlon,
                dtype=torch.float32,
                device=torch.device("cpu"),
                lat_order=str(lat_order),
                lon_origin=str(lon_origin),
            )
            self.register_buffer("coord_channels", cc, persistent=False)
        else:
            self.register_buffer("coord_channels", torch.zeros((0, nlat, nlon), dtype=torch.float32), persistent=False)

    def _param_maps(self, params: torch.Tensor) -> torch.Tensor:
        """Broadcast parameter vectors to per-pixel conditioning maps."""
        bsz, p = params.shape
        return params[:, :, None, None].expand(bsz, p, self.nlat, self.nlon)

    def rollout(self, params: torch.Tensor, *, steps: int) -> torch.Tensor:
        """Roll forward for ``steps`` iterations and return the final state."""
        if steps < 1:
            raise ValueError("rollout steps must be >= 1")
        if params.ndim != 2 or params.shape[-1] != self.param_dim:
            raise ValueError(f"params must be [B,{self.param_dim}], got {tuple(params.shape)}")

        state = self.ic(params)

        coord = self.coord_channels.to(device=params.device, dtype=params.dtype)
        if coord.numel():
            coord = coord.unsqueeze(0).expand(params.shape[0], -1, -1, -1)

        pmap = self._param_maps(params) if self.include_param_maps else None

        step_scale = 1.0 / float(int(steps))
        for _ in range(int(steps)):
            parts = [state]
            if pmap is not None:
                parts.append(pmap)
            if coord.numel():
                parts.append(coord)
            x = torch.cat(parts, dim=1)
            state = self.stepper(x, step_scale=step_scale)

        return state

    def forward(self, params: torch.Tensor, steps: int) -> torch.Tensor:
        """Alias for ``rollout`` to match standard module call signature."""
        return self.rollout(params, steps=int(steps))


class StateConditionedRolloutModel(nn.Module):
    """Autoregressive state-transition model with static conditioning channels."""

    def __init__(
        self,
        *,
        stepper: nn.Module,
        param_dim: int,
        state_chans: int,
        nlat: int,
        nlon: int,
        include_param_maps: bool,
        include_coord_channels: bool,
        lat_order: str = "north_to_south",
        lon_origin: str = "0_to_2pi",
    ) -> None:
        super().__init__()
        self.stepper = stepper
        self.param_dim = int(param_dim)
        self.state_chans = int(state_chans)
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.include_param_maps = bool(include_param_maps)
        self.include_coord_channels = bool(include_coord_channels)

        if self.include_coord_channels:
            cc = build_coord_channels_legendre_gauss(
                nlat,
                nlon,
                dtype=torch.float32,
                device=torch.device("cpu"),
                lat_order=str(lat_order),
                lon_origin=str(lon_origin),
            )
            self.register_buffer("coord_channels", cc, persistent=False)
        else:
            self.register_buffer("coord_channels", torch.zeros((0, nlat, nlon), dtype=torch.float32), persistent=False)

    def _param_maps(self, params: torch.Tensor) -> torch.Tensor:
        """Broadcast parameter vectors to per-pixel conditioning maps."""
        bsz, p = params.shape
        return params[:, :, None, None].expand(bsz, p, self.nlat, self.nlon)

    def rollout(self, state0: torch.Tensor, params: torch.Tensor, *, steps: int) -> torch.Tensor:
        """Roll forward from ``state0`` for ``steps`` iterations."""
        if steps < 1:
            raise ValueError("rollout steps must be >= 1")
        if state0.ndim != 4:
            raise ValueError(f"state0 must be [B,C,H,W], got {tuple(state0.shape)}")
        if state0.shape[1] != self.state_chans:
            raise ValueError(f"state0 channel count must be {self.state_chans}, got {state0.shape[1]}")
        if state0.shape[2] != self.nlat or state0.shape[3] != self.nlon:
            raise ValueError(
                f"state0 spatial shape must be ({self.nlat},{self.nlon}), got {tuple(state0.shape[-2:])}"
            )
        if params.ndim != 2 or params.shape[-1] != self.param_dim:
            raise ValueError(f"params must be [B,{self.param_dim}], got {tuple(params.shape)}")
        if int(state0.shape[0]) != int(params.shape[0]):
            raise ValueError(
                f"Batch size mismatch between state0 and params: {tuple(state0.shape)} vs {tuple(params.shape)}"
            )

        state = state0
        coord = self.coord_channels.to(device=params.device, dtype=params.dtype)
        if coord.numel():
            coord = coord.unsqueeze(0).expand(params.shape[0], -1, -1, -1)
        pmap = self._param_maps(params) if self.include_param_maps else None

        step_scale = 1.0 / float(int(steps))
        for _ in range(int(steps)):
            parts = [state]
            if pmap is not None:
                parts.append(pmap)
            if coord.numel():
                parts.append(coord)
            x = torch.cat(parts, dim=1)
            state = self.stepper(x, step_scale=step_scale)
        return state

    def forward(self, state0: torch.Tensor, params: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Alias for ``rollout`` to match standard module call signature."""
        return self.rollout(state0, params, steps=int(steps))


def build_rollout_model(
    *,
    img_size: tuple[int, int],
    state_chans: int,
    param_dim: int,
    cfg_model,
    lat_order: str = "north_to_south",
    lon_origin: str = "0_to_2pi",
) -> ParamRolloutModel:
    """Construct legacy parameter-only rollout model from shape and config."""
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
    _ensure_sfno_encoder_depth(
        base=base,
        in_chans=int(in_chans),
        encoder_layers=int(cfg_model.encoder_layers),
        mlp_ratio=float(cfg_model.mlp_ratio),
        bias=bool(cfg_model.bias),
    )

    stepper = ConditionalResidualWrapper(
        base,
        state_chans=state_chans,
        residual_prediction=bool(cfg_model.residual_prediction),
        residual_init_scale=float(cfg_model.residual_init_scale),
    )

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
        lat_order=str(lat_order),
        lon_origin=str(lon_origin),
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
        lat_order=str(lat_order),
        lon_origin=str(lon_origin),
    )


def build_state_conditioned_rollout_model(
    *,
    img_size: tuple[int, int],
    state_chans: int,
    param_dim: int,
    cfg_model,
    lat_order: str = "north_to_south",
    lon_origin: str = "0_to_2pi",
) -> StateConditionedRolloutModel:
    """Construct state-conditioned rollout model for transition supervision."""
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
    _ensure_sfno_encoder_depth(
        base=base,
        in_chans=int(in_chans),
        encoder_layers=int(cfg_model.encoder_layers),
        mlp_ratio=float(cfg_model.mlp_ratio),
        bias=bool(cfg_model.bias),
    )

    stepper = ConditionalResidualWrapper(
        base,
        state_chans=state_chans,
        residual_prediction=bool(cfg_model.residual_prediction),
        residual_init_scale=float(cfg_model.residual_init_scale),
    )

    return StateConditionedRolloutModel(
        stepper=stepper,
        param_dim=param_dim,
        state_chans=state_chans,
        nlat=h,
        nlon=w,
        include_param_maps=bool(cfg_model.include_param_maps),
        include_coord_channels=bool(cfg_model.include_coord_channels),
        lat_order=str(lat_order),
        lon_origin=str(lon_origin),
    )
