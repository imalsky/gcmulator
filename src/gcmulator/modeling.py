"""Model-building utilities for configurable spherical state-transition emulation."""

from __future__ import annotations

import math
import warnings
from typing import Any, List, Sequence

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
    """Import the SFNO model class with a small cross-version compatibility shim."""
    import torch_harmonics.examples.models.sfno as sfno_mod

    # Some torch-harmonics builds reference DropPath from ``sfno.py`` without
    # re-exporting it there. Patch the module once so downstream code can stay
    # version-agnostic.
    if not hasattr(sfno_mod, "DropPath"):
        from torch_harmonics.examples.models._layers import DropPath

        sfno_mod.DropPath = DropPath

    SphericalFourierNeuralOperator = sfno_mod.SphericalFourierNeuralOperator

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

    def __init__(self, nlat: int, nlon: int, grid: str) -> None:
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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute quadrature-weighted MSE on the sphere."""
        if pred.shape != target.shape:
            raise ValueError(
                f"shape mismatch pred={tuple(pred.shape)} "
                f"target={tuple(target.shape)}"
            )
        quad = self.quad
        if quad.ndim == 2:
            quad = quad[:, 0]
        quad = quad[None, None, :, None].to(device=pred.device, dtype=pred.dtype)
        return (
            (((pred - target) ** 2) * quad).sum(dim=(-2, -1))
        ).mean()


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
            # Start from identity FiLM modulation so the residual path controls
            # the initial behavior rather than arbitrary parameter scaling.
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


class StateConditionedTransitionModel(nn.Module):
    """Direct-jump transition model with FiLM-conditioned SFNO latent states."""

    def __init__(
        self,
        *,
        base: nn.Module,
        param_dim: int,
        input_state_chans: int,
        target_state_chans: int,
        residual_input_indices: Sequence[int],
        nlat: int,
        nlon: int,
        residual_prediction: bool,
        residual_init_scale: float,
    ) -> None:
        """Wrap the base SFNO with state/conditioning adapters and residual heads."""
        super().__init__()
        self.base = base
        self.param_dim = int(param_dim)
        self.input_state_chans = int(input_state_chans)
        self.target_state_chans = int(target_state_chans)
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.residual_prediction = bool(residual_prediction)
        self.conditioner = FiLMConditioner(
            param_dim=self.param_dim,
            embed_dim=int(getattr(base, "embed_dim")),
            num_sites=int(getattr(base, "num_layers")) + 1,
        )
        self.register_buffer(
            "residual_input_indices",
            torch.tensor([int(index) for index in residual_input_indices], dtype=torch.int64),
            persistent=False,
        )

        scale_init = torch.tensor(float(residual_init_scale), dtype=torch.float32)
        if self.residual_prediction:
            self.residual_scale = nn.Parameter(scale_init)
        else:
            self.register_buffer("residual_scale", scale_init, persistent=False)

    @staticmethod
    def _apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation to a latent feature map."""
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, state0: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Predict the next prognostic state autoregressively.

        Args:
            state0:
                Normalized prognostic state ``[B, C, H, W]`` where ``C`` is
                the number of prognostic fields (Phi, eta, delta).
            params:
                Normalized conditioning ``[B, P]`` (physical parameters +
                ``log10_transition_days``).

        Returns:
            Normalized prognostic state ``[B, C, H, W]``.
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
        x = self.base.encoder(state0)
        x = self._apply_film(x, gamma[:, 0], beta[:, 0])
        x = self.base.pos_embed(x)
        x = self.base.pos_drop(x)
        for block_index, block in enumerate(self.base.blocks):
            x = block(x)
            x = self._apply_film(x, gamma[:, block_index + 1], beta[:, block_index + 1])
        output = self.base.decoder(x)

        if not self.residual_prediction:
            return output
        residual = state0.index_select(
            dim=1,
            index=self.residual_input_indices.to(device=state0.device),
        )
        scale = self.residual_scale.to(device=output.device, dtype=output.dtype)
        return residual + scale * output


def build_state_conditioned_transition_model(
    *,
    img_size: tuple[int, int],
    input_state_chans: int,
    target_state_chans: int,
    param_dim: int,
    residual_input_indices: Sequence[int],
    cfg_model,
) -> StateConditionedTransitionModel:
    """Construct the configurable FiLM-conditioned SFNO transition model.

    The autoregressive architecture uses the same ``C`` prognostic channels
    (Phi, eta, delta) for both input and output.

    Args:
        img_size:
            Spatial grid size ``(H, W)``.
        input_state_chans:
            Input channel count (``3`` for the autoregressive contract).
        target_state_chans:
            Output channel count (``3`` for the autoregressive contract).
        param_dim:
            Conditioning width ``P`` (physical parameters +
            ``log10_transition_days``).

    Returns:
        A module implementing ``model(state0, conditioning) ->
        [B, C, H, W]``.
    """
    height, width = int(img_size[0]), int(img_size[1])
    in_chans = int(input_state_chans)

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
        residual_input_indices=residual_input_indices,
        nlat=height,
        nlon=width,
        residual_prediction=bool(cfg_model.residual_prediction),
        residual_init_scale=float(cfg_model.residual_init_scale),
    )
