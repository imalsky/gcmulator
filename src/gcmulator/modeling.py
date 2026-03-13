"""Model-building utilities for configurable spherical state-transition emulation."""

from __future__ import annotations

import math
import warnings
from typing import Any, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from .config import TORCH_HARMONICS_REQUIRED_VERSION


_TORCH_HARMONICS_VERSION_WARNED = False


def ensure_torch_harmonics_importable() -> None:
    """Require an importable ``torch_harmonics`` installation with the APIs we use."""
    global _TORCH_HARMONICS_VERSION_WARNED
    try:
        import torch_harmonics
    except Exception as exc:
        raise RuntimeError(
            "Could not import torch_harmonics. Install it before training or export."
        ) from exc

    version = str(getattr(torch_harmonics, "__version__", "unknown"))
    try:
        import_sfno()
        from torch_harmonics.examples.losses import get_quadrature_weights  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Installed torch_harmonics is missing the required SFNO/loss APIs. "
            "Use a compatible build such as torch-harmonics==0.8.1."
        )
    if version != TORCH_HARMONICS_REQUIRED_VERSION and not _TORCH_HARMONICS_VERSION_WARNED:
        warnings.warn(
            "torch_harmonics version differs from the preferred training baseline: "
            f"expected {TORCH_HARMONICS_REQUIRED_VERSION}, got {version}. "
            "Proceeding because the required APIs are available.",
            RuntimeWarning,
            stacklevel=2,
        )
        _TORCH_HARMONICS_VERSION_WARNED = True


def import_sfno() -> type[nn.Module]:
    """Import the SFNO model class."""
    from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator

    return SphericalFourierNeuralOperator


def _count_pointwise_convs(seq: nn.Module) -> int | None:
    """Count 1x1 ``Conv2d`` layers in an encoder block."""
    if not isinstance(seq, nn.Sequential):
        return None
    return sum(
        1
        for module in seq
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1)
    )


def _build_pointwise_stack(
    *,
    in_chans: int,
    out_chans: int,
    num_layers: int,
    hidden_dim: int,
    activation_fn: type[nn.Module],
    final_bias: bool,
) -> nn.Sequential:
    """Build a pointwise Conv2d stack used to patch SFNO encoder depth."""
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
    """Patch torch-harmonics 0.8.1 encoder depth, which otherwise ignores ``encoder_layers``."""
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
    """Resolve the runtime torch device from the config string.

    ``auto`` prefers CUDA, then CPU.
    """
    mode = str(device_cfg).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device='cuda' but CUDA is unavailable")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_context(device: torch.device, amp_mode: str) -> Any:
    """Return autocast context manager for configured AMP mode."""
    mode = str(amp_mode).lower()
    if mode == "bf16" and device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if mode == "fp16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


class SphereLoss(nn.Module):
    """Quadrature-weighted squared L2 over the sphere."""

    def __init__(
        self,
        nlat: int,
        nlon: int,
        grid: str,
        *,
        channel_weights: Sequence[float] | None = None,
    ) -> None:
        """Precompute quadrature weights for the chosen spherical grid."""
        super().__init__()
        from torch_harmonics.examples.losses import get_quadrature_weights

        weights = get_quadrature_weights(
            nlat=nlat,
            nlon=nlon,
            grid=grid,
            tile=False,
            normalized=True,
        )
        self.register_buffer("quad", weights.to(torch.float32))
        if channel_weights is None:
            self.register_buffer(
                "channel_weights",
                torch.empty((0,), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer(
                "channel_weights",
                torch.as_tensor(channel_weights, dtype=torch.float32),
            )

    def per_channel_losses(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute one quadrature-weighted MSE scalar per channel."""
        if pred.shape != target.shape:
            raise ValueError(
                f"shape mismatch pred={tuple(pred.shape)} "
                f"target={tuple(target.shape)}"
            )
        quad = self.quad
        if quad.ndim == 2:
            quad = quad[:, 0]
        quad = quad[None, None, :, None].to(device=pred.device, dtype=pred.dtype)
        return (((pred - target) ** 2) * quad).sum(dim=(-2, -1)).mean(dim=0)

    def reduce_channel_losses(self, per_channel_loss: torch.Tensor) -> torch.Tensor:
        """Aggregate per-channel losses with an optional weighted mean."""
        if per_channel_loss.ndim != 1:
            raise ValueError(
                f"per_channel_loss must be rank-1, got {tuple(per_channel_loss.shape)}"
            )
        if self.channel_weights.numel() == 0:
            return per_channel_loss.mean()
        if self.channel_weights.shape != per_channel_loss.shape:
            raise ValueError(
                "channel weight count must match the model output channels: "
                f"weights={tuple(self.channel_weights.shape)}, "
                f"channels={tuple(per_channel_loss.shape)}"
            )
        weights = self.channel_weights.to(
            device=per_channel_loss.device,
            dtype=per_channel_loss.dtype,
        )
        return torch.sum(per_channel_loss * weights) / torch.sum(weights)

    def loss_with_channels(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the aggregate loss and the ordered per-channel loss vector."""
        per_channel_loss = self.per_channel_losses(pred, target)
        return self.reduce_channel_losses(per_channel_loss), per_channel_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute quadrature-weighted MSE on the sphere."""
        loss, _ = self.loss_with_channels(pred, target)
        return loss


class FiLMConditioner(nn.Module):
    """Map global conditioning vectors into per-stage FiLM parameters."""

    def __init__(self, *, param_dim: int, embed_dim: int, num_sites: int) -> None:
        """Initialize the FiLM parameter MLP."""
        super().__init__()
        self.param_dim = int(param_dim)
        self.embed_dim = int(embed_dim)
        self.num_sites = int(num_sites)
        hidden_dim = int(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(self.param_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * self.num_sites * self.embed_dim),
        )
        last_layer = self.net[-1]
        if isinstance(last_layer, nn.Linear):
            # Start from identity FiLM modulation so conditioning only changes
            # behavior once the network learns a nonzero modulation.
            nn.init.constant_(last_layer.weight, 0.0)
            nn.init.constant_(last_layer.bias, 0.0)

    def forward(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FiLM scale and shift tensors with shape ``[B,S,C]``."""
        if params.ndim != 2 or int(params.shape[-1]) != self.param_dim:
            raise ValueError(f"params must be [B,{self.param_dim}], got {tuple(params.shape)}")
        raw = self.net(params)
        raw = raw.view(int(params.shape[0]), self.num_sites, 2, self.embed_dim)
        gamma = raw[:, :, 0]
        beta = raw[:, :, 1]
        return gamma, beta


def build_coord_channels_legendre_gauss(
    nlat: int,
    nlon: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    lat_order: str = "north_to_south",
    lon_origin: str = "0_to_2pi",
) -> torch.Tensor:
    """Build fixed sin/cos latitude-longitude channels on a Legendre-Gauss grid."""
    mu, _ = np.polynomial.legendre.leggauss(int(nlat))
    if lat_order == "north_to_south":
        mu = mu[::-1].copy()
    elif lat_order != "south_to_north":
        raise ValueError(f"Unsupported lat_order: {lat_order}")

    lat = np.arcsin(mu).astype(np.float32)
    if lon_origin == "0_to_2pi":
        lon = np.arange(int(nlon), dtype=np.float32) * ((2.0 * np.pi) / float(nlon))
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
    stacked = np.stack([sin_lat, cos_lat, sin_lon, cos_lon], axis=0).astype(np.float32)
    return torch.from_numpy(stacked).to(device=device, dtype=dtype)


class StateConditionedTransitionModel(nn.Module):
    """Direct-jump transition model with FiLM-conditioned SFNO latent states."""

    def __init__(
        self,
        *,
        base: nn.Module,
        param_dim: int,
        input_state_chans: int,
        target_state_chans: int,
        nlat: int,
        nlon: int,
        include_coord_channels: bool = False,
        residual_prediction: bool,
        lat_order: str = "north_to_south",
        lon_origin: str = "0_to_2pi",
    ) -> None:
        """Wrap the base SFNO with state/conditioning adapters and an optional big skip."""
        super().__init__()
        self.base = base
        self.param_dim = int(param_dim)
        self.input_state_chans = int(input_state_chans)
        self.target_state_chans = int(target_state_chans)
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.include_coord_channels = bool(include_coord_channels)
        self.residual_prediction = bool(residual_prediction)
        if self.residual_prediction and self.input_state_chans != self.target_state_chans:
            raise ValueError(
                "Fixed residual_prediction requires identical input and target channels"
            )
        self.conditioner = FiLMConditioner(
            param_dim=self.param_dim,
            embed_dim=int(getattr(base, "embed_dim")),
            num_sites=int(getattr(base, "num_layers")) + 1,
        )
        if self.include_coord_channels:
            channels = build_coord_channels_legendre_gauss(
                nlat=nlat,
                nlon=nlon,
                dtype=torch.float32,
                device=torch.device("cpu"),
                lat_order=str(lat_order),
                lon_origin=str(lon_origin),
            )
        else:
            channels = torch.zeros((0, nlat, nlon), dtype=torch.float32)
        self.register_buffer("coord_channels", channels, persistent=False)

    @staticmethod
    def _apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation to a latent feature map."""
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def _build_features(self, state0: torch.Tensor) -> torch.Tensor:
        """Concatenate static coordinate channels onto the visible input state."""
        pieces = [state0]
        coord = self.coord_channels.to(device=state0.device, dtype=state0.dtype)
        if coord.numel():
            pieces.append(coord.unsqueeze(0).expand(state0.shape[0], -1, -1, -1))
        return torch.cat(pieces, dim=1)

    def forward(self, state0: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Predict the next visible state autoregressively.

        Args:
            state0:
                Normalized visible state ``[B, C, H, W]`` where ``C`` is the
                number of stored state fields.
            params:
                Normalized conditioning ``[B, P]`` (physical parameters +
                ``log10_transition_days``).

        Returns:
            Normalized visible state ``[B, C, H, W]``.
        """
        if state0.ndim != 4:
            raise ValueError(f"state0 must be [B,C,H,W], got {tuple(state0.shape)}")
        if state0.shape[1] != self.input_state_chans:
            raise ValueError(f"state0 channel count must be {self.input_state_chans}")
        if state0.shape[2] != self.nlat or state0.shape[3] != self.nlon:
            raise ValueError(f"state0 spatial shape must be ({self.nlat},{self.nlon})")
        if params.ndim != 2 or params.shape[0] != state0.shape[0]:
            raise ValueError("params must be [B,P] and batch-aligned with state0")

        gamma, beta = self.conditioner(params)  # shape: [B, num_sites, embed_dim]
        x = self.base.encoder(self._build_features(state0))
        x = self._apply_film(x, gamma[:, 0], beta[:, 0])
        x = self.base.pos_embed(x)
        x = self.base.pos_drop(x)
        for block_index, block in enumerate(self.base.blocks):
            x = block(x)
            x = self._apply_film(x, gamma[:, block_index + 1], beta[:, block_index + 1])
        output = self.base.decoder(x)

        if not self.residual_prediction:
            return output
        return state0 + output


def build_state_conditioned_transition_model(
    *,
    img_size: tuple[int, int],
    input_state_chans: int,
    target_state_chans: int,
    param_dim: int,
    cfg_model,
    lat_order: str = "north_to_south",
    lon_origin: str = "0_to_2pi",
) -> StateConditionedTransitionModel:
    """Construct the configurable FiLM-conditioned SFNO transition model.

    The autoregressive architecture uses the same ``C`` visible state channels
    for both input and output.

    Args:
        img_size:
            Spatial grid size ``(H, W)``.
        input_state_chans:
            Input channel count for the visible-state contract.
        target_state_chans:
            Output channel count for the visible-state contract.
        param_dim:
            Conditioning width ``P`` (physical parameters +
            ``log10_transition_days``).

    Returns:
        A module implementing ``model(state0, conditioning) ->
        [B, C, H, W]``.
    """
    height, width = int(img_size[0]), int(img_size[1])
    coord_chans = 4 if bool(cfg_model.include_coord_channels) else 0
    in_chans = int(input_state_chans) + coord_chans

    SphericalFNO = import_sfno()
    base = SphericalFNO(
        img_size=(height, width),
        grid=str(cfg_model.grid),
        grid_internal=str(cfg_model.grid_internal),
        scale_factor=int(cfg_model.scale_factor),
        in_chans=int(in_chans),
        out_chans=int(target_state_chans),
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

    return StateConditionedTransitionModel(
        base=base,
        param_dim=int(param_dim),
        input_state_chans=int(input_state_chans),
        target_state_chans=int(target_state_chans),
        nlat=height,
        nlon=width,
        include_coord_channels=bool(cfg_model.include_coord_channels),
        residual_prediction=bool(cfg_model.residual_prediction),
        lat_order=str(lat_order),
        lon_origin=str(lon_origin),
    )
