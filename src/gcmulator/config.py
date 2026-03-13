"""Configuration schema, parsing, and validation for GCMulator.

The module keeps the runtime contract explicit: config files are parsed into
typed dataclasses, unknown keys are rejected early, and cross-field validation
is centralized in one place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# -----------------------------------------------------------------------------
# Global constants and schema enums
# -----------------------------------------------------------------------------
SECONDS_PER_DAY = 86400.0
MIN_TRANSITIONS = 1
PROBABILITY_MIN = 0.0
PROBABILITY_MAX = 1.0
DEFAULT_SCHEDULER_MIN_LR_RATIO = 50.0
DEFAULT_SCHEDULER_EPS = 1.0e-10

TransformName = Literal["none", "log10", "signed_log1p"]
PairSamplingPolicy = Literal["uniform_pairs", "uniform_gaps", "inverse_time"]
PairIterationMode = Literal["live_sampled_gpu", "resample_from_saved_sequences"]

PHYSICAL_STATE_FIELDS = ("Phi", "U", "V", "eta", "delta")
PROGNOSTIC_STATE_FIELDS = ("Phi", "eta", "delta")
CONDITIONING_PARAM_NAMES = (
    "a_m",
    "omega_rad_s",
    "Phibar",
    "DPhieq",
    "taurad_s",
    "taudrag_s",
    "g_m_s2",
)
TRANSITION_TIME_NAME = "log10_transition_days"
USER_CORE_PARAM_NAMES = (
    "a_m",
    "omega_rad_s",
    "Phibar",
    "DPhieq",
    "g_m_s2",
)
TAURAD_ALIASES = ("taurad_s", "taurad_hours")
TAUDRAG_ALIASES = ("taudrag_s", "taudrag_hours")
INTERNAL_FIXED_PARAM_NAMES = ("K6", "K6Phi")
VALID_PARAM_DISTS = {"uniform", "loguniform", "const", "fixed", "mixture_off_loguniform"}
TORCH_HARMONICS_REQUIRED_VERSION = "0.8.1"


def canonicalize_state_field(field_name: str) -> str:
    """Return the canonical visible-state field name."""
    candidate = str(field_name)
    if candidate not in PHYSICAL_STATE_FIELDS:
        raise ValueError(f"Unsupported state field: {field_name}")
    return candidate


@dataclass(frozen=True)
class Extended9Params:
    """MY_SWAMP run parameters (7 user-facing conditioning + 2 internal controls)."""

    a_m: float
    omega_rad_s: float
    Phibar: float
    DPhieq: float
    taurad_s: float
    taudrag_s: float
    g_m_s2: float
    K6: float
    K6Phi: Optional[float]

    def to_vector(self) -> List[float]:
        """Return the canonical user-facing conditioning vector."""
        return [
            float(self.a_m),
            float(self.omega_rad_s),
            float(self.Phibar),
            float(self.DPhieq),
            float(self.taurad_s),
            float(self.taudrag_s),
            float(self.g_m_s2),
        ]


@dataclass(frozen=True)
class NormalizationConfig:
    """Configurable per-channel transforms before z-score normalization."""

    field_transforms: Dict[str, TransformName] = field(default_factory=dict)
    zscore_eps: float = 1e-8
    log10_eps: float = 1e-30
    signed_log1p_scale: float = 1.0


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths for raw data, processed data, and model artifacts."""

    dataset_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    model_dir: str = "models/model_flow"
    overwrite_dataset: bool = False


@dataclass(frozen=True)
class SolverConfig:
    """MY_SWAMP integration controls used for data generation."""

    M: int = 42
    dt_seconds: float = 240.0
    default_time_days: float = 100.0
    starttime_index: int = 2


@dataclass(frozen=True)
class GeometryConfig:
    """Grid-orientation conventions applied before storing states."""

    flip_latitude_to_north_south: bool = True
    roll_longitude_to_0_2pi: bool = True


@dataclass(frozen=True)
class ParameterSpec:
    """One sampling rule for a single physical parameter."""

    name: str
    dist: str
    min: Optional[float] = None
    max: Optional[float] = None
    value: Optional[float] = None
    p_off: Optional[float] = None
    off_value: Optional[float] = None
    on_min: Optional[float] = None
    on_max: Optional[float] = None


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling strategy for raw simulation generation and live jump sampling.

    ``generation_workers`` is a trajectory batch size for vectorized JAX
    generation, not an OS worker pool.  ``0`` selects the backend-aware auto
    policy used by the generation entrypoint.

    Raw generation stores one uniformly spaced checkpoint sequence per
    simulation. Training then samples variable-``dt`` direct-jump pairs live
    from those saved checkpoints on GPU.

    Setting ``burn_in_days`` to 0 allows anchor times at the very start of
    the simulation (right after the numerical warm-up controlled by
    ``solver.starttime_index``).  This is essential for training the model
    to evolve an arbitrary initial state toward equilibrium.
    """

    seed: int = 0
    n_sims: int = 500
    generation_workers: int = 0
    burn_in_days: float = 0.0
    saved_checkpoint_interval_days: float = 1.0
    saved_snapshots_per_sim: Optional[int] = None
    fixed_transition_steps: Optional[int] = None
    live_pairs_per_sequence: int = 10
    pairs_per_sim: int = 10
    pair_sampling_policy: PairSamplingPolicy = "inverse_time"
    live_transition_days_min: float = 0.1
    live_transition_days_max: float = 100.0
    live_transition_tolerance_fraction: float = 0.1
    parameters: List[ParameterSpec] = field(default_factory=list)

    def min_live_transition_days(self) -> float:
        """Return the minimum live-sampled transition duration in physical days."""
        return float(self.live_transition_days_min)

    def max_live_transition_days(self) -> float:
        """Return the maximum live-sampled transition duration in physical days."""
        return float(self.live_transition_days_max)

    def uses_variable_live_transition(self) -> bool:
        """Return True when live transition duration is sampled from a non-degenerate range."""
        return self.max_live_transition_days() > self.min_live_transition_days()


@dataclass(frozen=True)
class ParamNormConfig:
    """Normalization settings for the conditioning parameter vector."""

    mode: str = "zscore"
    eps: float = 1e-8


@dataclass(frozen=True)
class NormalizationSettings:
    """Container for state and parameter normalization settings."""

    state: NormalizationConfig = field(default_factory=NormalizationConfig)
    params: ParamNormConfig = field(default_factory=ParamNormConfig)


@dataclass(frozen=True)
class ModelConfig:
    """State-conditioned SFNO controls."""

    grid: str = "legendre-gauss"
    grid_internal: str = "legendre-gauss"
    scale_factor: int = 2
    embed_dim: int = 128
    num_layers: int = 6
    encoder_layers: int = 2
    activation_function: str = "gelu"
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.05
    normalization_layer: str = "instance_norm"
    hard_thresholding_fraction: float = 1.0
    residual_prediction: bool = True
    include_coord_channels: bool = False
    pos_embed: str = "spectral"
    bias: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate scheduler hyperparameters."""

    type: str = "plateau"
    warmup_epochs: int = 10
    factor: float = 0.5
    patience: int = 10
    min_lr: float = 0.0
    eps: float = DEFAULT_SCHEDULER_EPS


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization, data-loader, and split settings for training."""

    seed: int = 0
    device: str = "auto"
    amp_mode: str = "none"
    deterministic: bool = False
    epochs: int = 50
    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = True
    preload_to_gpu: bool = True
    pair_iteration_mode: PairIterationMode = "live_sampled_gpu"
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    channel_loss_weights: Optional[Dict[str, float]] = None
    grad_clip_norm: float = 1.0
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass(frozen=True)
class GCMulatorConfig:
    """Top-level runtime configuration object."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    normalization: NormalizationSettings = field(default_factory=NormalizationSettings)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


TOP_LEVEL_CONFIG_KEYS = {
    "paths",
    "solver",
    "geometry",
    "sampling",
    "normalization",
    "model",
    "training",
}
PATHS_KEYS = {"dataset_dir", "processed_dir", "model_dir", "overwrite_dataset"}
SOLVER_KEYS = {"M", "dt_seconds", "default_time_days", "starttime_index"}
GEOMETRY_KEYS = {"flip_latitude_to_north_south", "roll_longitude_to_0_2pi"}
SAMPLING_KEYS = {
    "seed",
    "n_sims",
    "generation_workers",
    "burn_in_days",
    "saved_checkpoint_interval_days",
    "saved_snapshots_per_sim",
    "fixed_transition_steps",
    "live_pairs_per_sequence",
    "pairs_per_sim",
    "pair_sampling_policy",
    "live_transition_days_min",
    "live_transition_days_max",
    "live_transition_tolerance_fraction",
    "parameters",
}
PARAM_SPEC_KEYS = {"name", "dist", "min", "max", "value", "p_off", "off_value", "on_min", "on_max"}
NORMALIZATION_KEYS = {"state", "params"}
NORM_STATE_KEYS = {"field_transforms", "zscore_eps", "log10_eps", "signed_log1p_scale"}
NORM_PARAMS_KEYS = {"mode", "eps"}
MODEL_KEYS = {
    "grid",
    "grid_internal",
    "scale_factor",
    "embed_dim",
    "num_layers",
    "encoder_layers",
    "activation_function",
    "use_mlp",
    "mlp_ratio",
    "drop_rate",
    "drop_path_rate",
    "normalization_layer",
    "hard_thresholding_fraction",
    "residual_prediction",
    "include_coord_channels",
    "pos_embed",
    "bias",
}
TRAINING_KEYS = {
    "seed",
    "device",
    "amp_mode",
    "deterministic",
    "epochs",
    "batch_size",
    "num_workers",
    "shuffle",
    "preload_to_gpu",
    "pair_iteration_mode",
    "learning_rate",
    "weight_decay",
    "channel_loss_weights",
    "grad_clip_norm",
    "val_fraction",
    "test_fraction",
    "split_seed",
    "scheduler",
}
SCHEDULER_KEYS = {"type", "warmup_epochs", "factor", "patience", "min_lr", "eps"}


def _load_raw_config(config_path: Path) -> Dict[str, Any]:
    """Load a JSON or YAML config from disk."""
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("YAML config requested but PyYAML is not installed") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON/YAML object")
    return data


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _parse_bool(value: Any, *, field_name: str) -> bool:
    """Parse strict boolean config fields."""
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean, got {value!r}")


def _parse_optional_int(value: Any, *, field_name: str) -> Optional[int]:
    """Parse optional integer config fields without silent coercion."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer or null, got {value!r}")
    return int(value)


def _parse_paths(d: Dict[str, Any]) -> PathsConfig:
    """Parse the ``paths`` section."""
    return PathsConfig(
        dataset_dir=str(d.get("dataset_dir", "data/raw")),
        processed_dir=str(d.get("processed_dir", "data/processed")),
        model_dir=str(d.get("model_dir", "models/model_flow")),
        overwrite_dataset=_parse_bool(
            d.get("overwrite_dataset", False),
            field_name="paths.overwrite_dataset",
        ),
    )


def _parse_solver(d: Dict[str, Any]) -> SolverConfig:
    """Parse the ``solver`` section."""
    return SolverConfig(
        M=int(d.get("M", 42)),
        dt_seconds=float(d.get("dt_seconds", 240.0)),
        default_time_days=float(d.get("default_time_days", 100.0)),
        starttime_index=int(d.get("starttime_index", 2)),
    )


def _parse_geometry(d: Dict[str, Any]) -> GeometryConfig:
    """Parse the ``geometry`` section."""
    return GeometryConfig(
        flip_latitude_to_north_south=_parse_bool(
            d.get("flip_latitude_to_north_south", True),
            field_name="geometry.flip_latitude_to_north_south",
        ),
        roll_longitude_to_0_2pi=_parse_bool(
            d.get("roll_longitude_to_0_2pi", True),
            field_name="geometry.roll_longitude_to_0_2pi",
        ),
    )


def _parse_parameter_spec(d: Dict[str, Any]) -> ParameterSpec:
    """Parse one entry from ``sampling.parameters``."""
    return ParameterSpec(
        name=str(d["name"]),
        dist=str(d["dist"]),
        min=None if d.get("min") is None else float(d["min"]),
        max=None if d.get("max") is None else float(d["max"]),
        value=None if d.get("value") is None else float(d["value"]),
        p_off=None if d.get("p_off") is None else float(d["p_off"]),
        off_value=None if d.get("off_value") is None else float(d["off_value"]),
        on_min=None if d.get("on_min") is None else float(d["on_min"]),
        on_max=None if d.get("on_max") is None else float(d["on_max"]),
    )


def _resolve_saved_checkpoint_interval_days(
    *,
    time_days: float,
    dt_seconds: float,
    saved_checkpoint_interval_days: float | None,
    saved_snapshots_per_sim: int | None,
) -> float:
    """Resolve one concrete checkpoint interval from either interval-days or snapshot count."""
    if saved_snapshots_per_sim is not None:
        if saved_snapshots_per_sim < 1:
            raise ValueError("sampling.saved_snapshots_per_sim must be >= 1")
        n_steps_total = max(
            1,
            int(round(float(time_days) * SECONDS_PER_DAY / float(dt_seconds))),
        )
        if int(saved_snapshots_per_sim) > int(n_steps_total):
            raise ValueError(
                "sampling.saved_snapshots_per_sim must be <= the total number of "
                f"solver steps ({n_steps_total})"
            )
        if int(n_steps_total) % int(saved_snapshots_per_sim) != 0:
            raise ValueError(
                "sampling.saved_snapshots_per_sim must divide the total number of "
                f"solver steps exactly; total_steps={n_steps_total}, "
                f"saved_snapshots_per_sim={saved_snapshots_per_sim}"
            )
        interval_steps = int(n_steps_total) // int(saved_snapshots_per_sim)
        return float(interval_steps) * float(dt_seconds) / float(SECONDS_PER_DAY)

    if saved_checkpoint_interval_days is None:
        return 1.0
    return float(saved_checkpoint_interval_days)


def _resolve_checkpoint_interval_steps(
    *,
    dt_seconds: float,
    saved_checkpoint_interval_days: float,
) -> int:
    """Resolve the saved checkpoint cadence onto the discrete solver-step grid."""
    return max(
        1,
        int(round(float(saved_checkpoint_interval_days) * SECONDS_PER_DAY / float(dt_seconds))),
    )


def _total_solver_steps(*, time_days: float, dt_seconds: float) -> int:
    """Resolve the total integration horizon onto the discrete solver-step grid."""
    return max(
        1,
        int(round(float(time_days) * SECONDS_PER_DAY / float(dt_seconds))),
    )


def _max_post_burn_in_gap_steps(
    *,
    time_days: float,
    dt_seconds: float,
    burn_in_days: float,
    saved_checkpoint_interval_days: float,
) -> int:
    """Return the largest checkpoint-to-checkpoint gap available after burn-in."""
    interval_steps = _resolve_checkpoint_interval_steps(
        dt_seconds=float(dt_seconds),
        saved_checkpoint_interval_days=float(saved_checkpoint_interval_days),
    )
    n_steps_total = _total_solver_steps(
        time_days=float(time_days),
        dt_seconds=float(dt_seconds),
    )
    last_checkpoint_step = (n_steps_total // interval_steps) * interval_steps
    burn_in_intervals = int(
        math.ceil(
            (
                float(burn_in_days)
                * SECONDS_PER_DAY
                / float(dt_seconds)
                / float(interval_steps)
            )
            - 1.0e-12
        )
    )
    first_anchor_checkpoint_step = burn_in_intervals * interval_steps
    return max(0, int(last_checkpoint_step - first_anchor_checkpoint_step))


def _parse_sampling(
    d: Dict[str, Any],
    *,
    solver: SolverConfig,
    raw_sampling: Optional[Dict[str, Any]] = None,
) -> SamplingConfig:
    """Parse the ``sampling`` section."""
    params_raw = d.get("parameters", [])
    if not isinstance(params_raw, list):
        raise ValueError("sampling.parameters must be a list")
    raw_sampling = raw_sampling or {}
    raw_has_saved_interval = (
        "saved_checkpoint_interval_days" in raw_sampling
        and raw_sampling.get("saved_checkpoint_interval_days") is not None
    )
    raw_has_saved_snapshots = (
        "saved_snapshots_per_sim" in raw_sampling
        and raw_sampling.get("saved_snapshots_per_sim") is not None
    )
    raw_has_fixed_transition_steps = (
        "fixed_transition_steps" in raw_sampling
        and raw_sampling.get("fixed_transition_steps") is not None
    )
    raw_has_live_transition_min = (
        "live_transition_days_min" in raw_sampling
        and raw_sampling.get("live_transition_days_min") is not None
    )
    raw_has_live_transition_max = (
        "live_transition_days_max" in raw_sampling
        and raw_sampling.get("live_transition_days_max") is not None
    )
    if raw_has_saved_interval and raw_has_saved_snapshots:
        raise ValueError(
            "sampling.saved_checkpoint_interval_days and "
            "sampling.saved_snapshots_per_sim are mutually exclusive"
        )
    if raw_has_fixed_transition_steps and (
        raw_has_live_transition_min or raw_has_live_transition_max
    ):
        raise ValueError(
            "sampling.fixed_transition_steps is mutually exclusive with "
            "sampling.live_transition_days_min/max"
        )
    saved_snapshots_per_sim = (
        None
        if d.get("saved_snapshots_per_sim") is None
        else int(d.get("saved_snapshots_per_sim"))
    )
    fixed_transition_steps = _parse_optional_int(
        d.get("fixed_transition_steps"),
        field_name="sampling.fixed_transition_steps",
    )
    resolved_interval_days = _resolve_saved_checkpoint_interval_days(
        time_days=float(solver.default_time_days),
        dt_seconds=float(solver.dt_seconds),
        saved_checkpoint_interval_days=(
            None
            if raw_has_saved_snapshots and not raw_has_saved_interval
            else float(d.get("saved_checkpoint_interval_days", 1.0))
        ),
        saved_snapshots_per_sim=saved_snapshots_per_sim,
    )
    if fixed_transition_steps is None:
        live_transition_days_min = float(d.get("live_transition_days_min", 0.1))
        live_transition_days_max = float(d.get("live_transition_days_max", 100.0))
    else:
        fixed_transition_days = (
            float(fixed_transition_steps) * float(solver.dt_seconds) / SECONDS_PER_DAY
        )
        live_transition_days_min = fixed_transition_days
        live_transition_days_max = fixed_transition_days
    return SamplingConfig(
        seed=int(d.get("seed", 0)),
        n_sims=int(d.get("n_sims", 500)),
        generation_workers=int(d.get("generation_workers", 0)),
        burn_in_days=float(d.get("burn_in_days", 0.0)),
        saved_checkpoint_interval_days=float(resolved_interval_days),
        saved_snapshots_per_sim=saved_snapshots_per_sim,
        fixed_transition_steps=fixed_transition_steps,
        live_pairs_per_sequence=int(d.get("live_pairs_per_sequence", 10)),
        pairs_per_sim=int(d.get("pairs_per_sim", d.get("live_pairs_per_sequence", 10))),
        pair_sampling_policy=str(d.get("pair_sampling_policy", "inverse_time")),
        live_transition_days_min=live_transition_days_min,
        live_transition_days_max=live_transition_days_max,
        live_transition_tolerance_fraction=float(
            d.get("live_transition_tolerance_fraction", 0.1)
        ),
        parameters=[_parse_parameter_spec(item) for item in params_raw],
    )


def _parse_norm(d: Dict[str, Any]) -> NormalizationSettings:
    """Parse the ``normalization`` section."""
    state = d.get("state", {})
    params = d.get("params", {})
    if not isinstance(state, dict):
        raise ValueError("normalization.state must be an object")
    if not isinstance(params, dict):
        raise ValueError("normalization.params must be an object")

    field_transforms_raw = state.get("field_transforms", {})
    if not isinstance(field_transforms_raw, dict):
        raise ValueError("normalization.state.field_transforms must be an object")
    field_transforms = {
        canonicalize_state_field(str(key)): str(value)
        for key, value in field_transforms_raw.items()
    }

    return NormalizationSettings(
        state=NormalizationConfig(
            field_transforms=field_transforms,
            zscore_eps=float(state.get("zscore_eps", 1e-8)),
            log10_eps=float(state.get("log10_eps", 1e-30)),
            signed_log1p_scale=float(state.get("signed_log1p_scale", 1.0)),
        ),
        params=ParamNormConfig(
            mode=str(params.get("mode", "zscore")),
            eps=float(params.get("eps", 1e-8)),
        ),
    )


def _parse_model(d: Dict[str, Any]) -> ModelConfig:
    """Parse the ``model`` section."""
    return ModelConfig(
        grid=str(d.get("grid", "legendre-gauss")),
        grid_internal=str(d.get("grid_internal", "legendre-gauss")),
        scale_factor=int(d.get("scale_factor", 2)),
        embed_dim=int(d.get("embed_dim", 128)),
        num_layers=int(d.get("num_layers", 6)),
        encoder_layers=int(d.get("encoder_layers", 2)),
        activation_function=str(d.get("activation_function", "gelu")),
        use_mlp=_parse_bool(d.get("use_mlp", True), field_name="model.use_mlp"),
        mlp_ratio=float(d.get("mlp_ratio", 2.0)),
        drop_rate=float(d.get("drop_rate", 0.0)),
        drop_path_rate=float(d.get("drop_path_rate", 0.05)),
        normalization_layer=str(d.get("normalization_layer", "instance_norm")),
        hard_thresholding_fraction=float(d.get("hard_thresholding_fraction", 1.0)),
        residual_prediction=_parse_bool(
            d.get("residual_prediction", True),
            field_name="model.residual_prediction",
        ),
        include_coord_channels=_parse_bool(
            d.get("include_coord_channels", False),
            field_name="model.include_coord_channels",
        ),
        pos_embed=str(d.get("pos_embed", "spectral")),
        bias=_parse_bool(d.get("bias", False), field_name="model.bias"),
    )


def _parse_scheduler(
    d: Dict[str, Any],
    *,
    learning_rate: float,
    raw_scheduler: Optional[Dict[str, Any]] = None,
) -> SchedulerConfig:
    """Parse the optional scheduler subsection."""
    scheduler_type = str(d.get("type", "plateau"))
    raw_scheduler = raw_scheduler or {}
    if "min_lr" in raw_scheduler:
        min_lr = float(raw_scheduler["min_lr"])
    elif scheduler_type == "none":
        min_lr = 0.0
    else:
        min_lr = float(learning_rate) / float(DEFAULT_SCHEDULER_MIN_LR_RATIO)
    return SchedulerConfig(
        type=scheduler_type,
        warmup_epochs=int(d.get("warmup_epochs", 10)),
        factor=float(d.get("factor", 0.5)),
        patience=int(d.get("patience", 10)),
        min_lr=min_lr,
        eps=float(d.get("eps", DEFAULT_SCHEDULER_EPS)),
    )


def _parse_training(
    d: Dict[str, Any],
    *,
    raw_training: Optional[Dict[str, Any]] = None,
) -> TrainingConfig:
    """Parse the ``training`` section."""
    scheduler_raw = d.get("scheduler", {})
    if not isinstance(scheduler_raw, dict):
        raise ValueError("training.scheduler must be an object")
    raw_training = raw_training or {}
    raw_scheduler = raw_training.get("scheduler", {})
    if raw_scheduler is None:
        raw_scheduler = {}
    if not isinstance(raw_scheduler, dict):
        raise ValueError("training.scheduler must be an object")
    raw_channel_loss_weights = d.get("channel_loss_weights")
    if raw_channel_loss_weights is None:
        channel_loss_weights = None
    else:
        if not isinstance(raw_channel_loss_weights, dict):
            raise ValueError("training.channel_loss_weights must be an object")
        channel_loss_weights = {
            canonicalize_state_field(str(key)): float(value)
            for key, value in raw_channel_loss_weights.items()
        }
    learning_rate = float(d.get("learning_rate", 3e-4))
    return TrainingConfig(
        seed=int(d.get("seed", 0)),
        device=str(d.get("device", "auto")),
        amp_mode=str(d.get("amp_mode", "none")),
        deterministic=_parse_bool(
            d.get("deterministic", False),
            field_name="training.deterministic",
        ),
        epochs=int(d.get("epochs", 50)),
        batch_size=int(d.get("batch_size", 8)),
        num_workers=int(d.get("num_workers", 0)),
        shuffle=_parse_bool(d.get("shuffle", True), field_name="training.shuffle"),
        preload_to_gpu=_parse_bool(
            d.get("preload_to_gpu", True),
            field_name="training.preload_to_gpu",
        ),
        pair_iteration_mode=str(d.get("pair_iteration_mode", "live_sampled_gpu")),
        learning_rate=learning_rate,
        weight_decay=float(d.get("weight_decay", 0.0)),
        channel_loss_weights=channel_loss_weights,
        grad_clip_norm=float(d.get("grad_clip_norm", 1.0)),
        val_fraction=float(d.get("val_fraction", 0.1)),
        test_fraction=float(d.get("test_fraction", 0.1)),
        split_seed=int(d.get("split_seed", 0)),
        scheduler=_parse_scheduler(
            scheduler_raw,
            learning_rate=learning_rate,
            raw_scheduler=raw_scheduler,
        ),
    )


def load_config(config_path: Path) -> GCMulatorConfig:
    """Load, merge with defaults, and validate an experiment config."""
    raw = _load_raw_config(config_path)
    _validate_raw_config_keys(raw)

    defaults = asdict(GCMulatorConfig())
    merged = _merge_dicts(defaults, raw)

    cfg = GCMulatorConfig(
        paths=_parse_paths(merged.get("paths", {})),
        solver=_parse_solver(merged.get("solver", {})),
        geometry=_parse_geometry(merged.get("geometry", {})),
        sampling=_parse_sampling(
            merged.get("sampling", {}),
            solver=_parse_solver(merged.get("solver", {})),
            raw_sampling=raw.get("sampling", {}),
        ),
        normalization=_parse_norm(merged.get("normalization", {})),
        model=_parse_model(merged.get("model", {})),
        training=_parse_training(
            merged.get("training", {}),
            raw_training=raw.get("training", {}),
        ),
    )
    validate_config(cfg)
    return cfg


def resolve_path(config_path: Path, path_value: str) -> Path:
    """Resolve a config-relative path into an absolute path."""
    return (config_path.parent / path_value).resolve()


def validate_config(cfg: GCMulatorConfig) -> None:
    """Validate cross-field constraints and supported option domains."""
    if cfg.solver.M not in (42, 63, 106):
        raise ValueError("solver.M must be one of [42, 63, 106]")
    if cfg.solver.dt_seconds <= 0:
        raise ValueError("solver.dt_seconds must be > 0")
    if cfg.solver.default_time_days <= 0:
        raise ValueError("solver.default_time_days must be > 0")
    if cfg.solver.starttime_index < 2:
        raise ValueError("solver.starttime_index must be >= 2")

    if cfg.sampling.n_sims < 1:
        raise ValueError("sampling.n_sims must be >= 1")
    if cfg.sampling.generation_workers < 0:
        raise ValueError("sampling.generation_workers must be >= 0")
    if cfg.sampling.burn_in_days < 0:
        raise ValueError("sampling.burn_in_days must be >= 0")
    if (
        cfg.sampling.saved_snapshots_per_sim is not None
        and cfg.sampling.saved_snapshots_per_sim < 1
    ):
        raise ValueError("sampling.saved_snapshots_per_sim must be >= 1")
    if cfg.sampling.saved_checkpoint_interval_days <= 0:
        raise ValueError("sampling.saved_checkpoint_interval_days must be > 0")
    if (
        cfg.sampling.fixed_transition_steps is not None
        and cfg.sampling.fixed_transition_steps < 1
    ):
        raise ValueError("sampling.fixed_transition_steps must be >= 1")
    if cfg.sampling.live_pairs_per_sequence < MIN_TRANSITIONS:
        raise ValueError(f"sampling.live_pairs_per_sequence must be >= {MIN_TRANSITIONS}")
    if cfg.sampling.pairs_per_sim < MIN_TRANSITIONS:
        raise ValueError(f"sampling.pairs_per_sim must be >= {MIN_TRANSITIONS}")
    if cfg.sampling.pair_sampling_policy not in {
        "uniform_pairs",
        "uniform_gaps",
        "inverse_time",
    }:
        raise ValueError(
            "sampling.pair_sampling_policy must be one of "
            "['uniform_pairs','uniform_gaps','inverse_time']"
        )
    if cfg.sampling.live_transition_days_min <= 0:
        raise ValueError("sampling.live_transition_days_min must be > 0")
    if cfg.sampling.live_transition_days_max <= 0:
        raise ValueError("sampling.live_transition_days_max must be > 0")
    if cfg.sampling.live_transition_days_min > cfg.sampling.live_transition_days_max:
        raise ValueError(
            "sampling.live_transition_days_min must be <= "
            "sampling.live_transition_days_max"
        )
    if not (0.0 <= cfg.sampling.live_transition_tolerance_fraction <= 1.0):
        raise ValueError("sampling.live_transition_tolerance_fraction must be in [0,1]")
    _validate_parameter_specs(cfg.sampling.parameters)

    burn_in_days = float(cfg.sampling.burn_in_days)
    min_jump_days = float(cfg.sampling.live_transition_days_min)
    available_days = float(cfg.solver.default_time_days) - burn_in_days
    if available_days < min_jump_days:
        raise ValueError(
            "No valid post-burn-in transition window exists: "
            f"available_days={available_days:.2f}, "
            f"min_jump_days={min_jump_days:.2f}"
        )
    if cfg.sampling.saved_checkpoint_interval_days > float(cfg.solver.default_time_days):
        raise ValueError(
            "sampling.saved_checkpoint_interval_days must be <= solver.default_time_days"
        )
    if cfg.sampling.fixed_transition_steps is not None:
        interval_steps = _resolve_checkpoint_interval_steps(
            dt_seconds=float(cfg.solver.dt_seconds),
            saved_checkpoint_interval_days=float(cfg.sampling.saved_checkpoint_interval_days),
        )
        if int(cfg.sampling.fixed_transition_steps) % int(interval_steps) != 0:
            raise ValueError(
                "sampling.fixed_transition_steps must be exactly representable by "
                "the saved checkpoint cadence"
            )
        max_gap_steps = _max_post_burn_in_gap_steps(
            time_days=float(cfg.solver.default_time_days),
            dt_seconds=float(cfg.solver.dt_seconds),
            burn_in_days=float(cfg.sampling.burn_in_days),
            saved_checkpoint_interval_days=float(cfg.sampling.saved_checkpoint_interval_days),
        )
        if int(cfg.sampling.fixed_transition_steps) > int(max_gap_steps):
            raise ValueError(
                "sampling.fixed_transition_steps must fit inside the post-burn-in "
                "saved checkpoint window"
            )

    if cfg.normalization.state.zscore_eps <= 0:
        raise ValueError("normalization.state.zscore_eps must be > 0")
    if cfg.normalization.state.log10_eps <= 0:
        raise ValueError("normalization.state.log10_eps must be > 0")
    if cfg.normalization.state.signed_log1p_scale <= 0:
        raise ValueError("normalization.state.signed_log1p_scale must be > 0")
    if cfg.normalization.params.mode not in {"zscore", "none"}:
        raise ValueError("normalization.params.mode must be one of ['zscore','none']")
    if cfg.normalization.params.eps <= 0:
        raise ValueError("normalization.params.eps must be > 0")
    unknown_transform_keys = sorted(
        set(cfg.normalization.state.field_transforms).difference(
            PHYSICAL_STATE_FIELDS
        )
    )
    if unknown_transform_keys:
        raise ValueError(
            "normalization.state.field_transforms contains unsupported field names: "
            f"{unknown_transform_keys}"
        )
    for field_name, transform in cfg.normalization.state.field_transforms.items():
        canonicalize_state_field(field_name)
        if transform not in {"none", "log10", "signed_log1p"}:
            raise ValueError(f"Invalid transform for {field_name}: {transform}")

    if cfg.training.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be one of ['auto','cpu','cuda']")
    if cfg.training.amp_mode not in {"none", "bf16", "fp16"}:
        raise ValueError("training.amp_mode must be one of ['none','bf16','fp16']")
    if cfg.training.pair_iteration_mode not in {
        "live_sampled_gpu",
        "resample_from_saved_sequences",
    }:
        raise ValueError(
            "training.pair_iteration_mode must be one of "
            "['live_sampled_gpu','resample_from_saved_sequences']"
        )
    if cfg.training.epochs < 1:
        raise ValueError("training.epochs must be >= 1")
    if cfg.training.batch_size < 1:
        raise ValueError("training.batch_size must be >= 1")
    if cfg.training.num_workers < 0:
        raise ValueError("training.num_workers must be >= 0")
    if cfg.training.pair_iteration_mode == "live_sampled_gpu":
        if not cfg.training.preload_to_gpu:
            raise ValueError("training.preload_to_gpu must be true for live GPU sampling")
        if int(cfg.training.batch_size) % int(cfg.sampling.live_pairs_per_sequence) != 0:
            raise ValueError(
                "training.batch_size must be divisible by "
                "sampling.live_pairs_per_sequence"
            )
        if cfg.training.num_workers != 0:
            raise ValueError("training.num_workers must be 0 when training.preload_to_gpu=true")
    else:
        if cfg.training.num_workers != 0:
            raise ValueError(
                "training.num_workers must be 0 for resample_from_saved_sequences"
            )
    if cfg.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be > 0")
    if cfg.training.weight_decay < 0:
        raise ValueError("training.weight_decay must be >= 0")
    if not (0.0 < cfg.training.val_fraction < 1.0):
        raise ValueError("training.val_fraction must be in (0,1)")
    if not (0.0 < cfg.training.test_fraction < 1.0):
        raise ValueError("training.test_fraction must be in (0,1)")
    if (cfg.training.val_fraction + cfg.training.test_fraction) >= 1.0:
        raise ValueError("training.val_fraction + training.test_fraction must be < 1")
    if cfg.training.scheduler.type not in {"cosine_warmup", "plateau", "none"}:
        raise ValueError(
            "training.scheduler.type must be one of "
            "['cosine_warmup','plateau','none']"
        )
    if cfg.training.scheduler.warmup_epochs < 0:
        raise ValueError("training.scheduler.warmup_epochs must be >= 0")
    if cfg.training.scheduler.factor <= 0 or cfg.training.scheduler.factor >= 1:
        raise ValueError("training.scheduler.factor must be in (0,1)")
    if cfg.training.scheduler.patience < 0:
        raise ValueError("training.scheduler.patience must be >= 0")
    if cfg.training.scheduler.min_lr < 0:
        raise ValueError("training.scheduler.min_lr must be >= 0")
    if cfg.training.scheduler.min_lr > cfg.training.learning_rate:
        raise ValueError("training.scheduler.min_lr must be <= training.learning_rate")
    if cfg.training.scheduler.eps <= 0:
        raise ValueError("training.scheduler.eps must be > 0")

    if cfg.model.grid != "legendre-gauss":
        raise ValueError("model.grid must be 'legendre-gauss'")
    if cfg.model.grid_internal != "legendre-gauss":
        raise ValueError("model.grid_internal must be 'legendre-gauss'")
    if cfg.model.grid_internal != cfg.model.grid:
        raise ValueError("model.grid_internal must match model.grid")
    if cfg.model.scale_factor < 1:
        raise ValueError("model.scale_factor must be >= 1")
    if cfg.model.embed_dim < 1:
        raise ValueError("model.embed_dim must be >= 1")
    if cfg.model.num_layers < 1:
        raise ValueError("model.num_layers must be >= 1")
    if cfg.model.encoder_layers < 1:
        raise ValueError("model.encoder_layers must be >= 1")
    if cfg.model.activation_function not in {"relu", "gelu", "identity"}:
        raise ValueError("model.activation_function must be one of ['relu','gelu','identity']")
    if cfg.model.mlp_ratio <= 0:
        raise ValueError("model.mlp_ratio must be > 0")
    if not (0 <= cfg.model.drop_rate < 1):
        raise ValueError("model.drop_rate must be in [0,1)")
    if not (0 <= cfg.model.drop_path_rate < 1):
        raise ValueError("model.drop_path_rate must be in [0,1)")
    if cfg.model.normalization_layer not in {"none", "layer_norm", "instance_norm"}:
        raise ValueError(
            "model.normalization_layer must be one of "
            "['none','layer_norm','instance_norm']"
        )
    if not (0 < cfg.model.hard_thresholding_fraction <= 1):
        raise ValueError("model.hard_thresholding_fraction must be in (0,1]")
    if cfg.model.pos_embed not in {
        "none",
        "sequence",
        "spectral",
        "learnable lat",
        "learnable latlon",
    }:
        raise ValueError(
            "model.pos_embed must be one of "
            "['none','sequence','spectral','learnable lat','learnable latlon']"
        )

    if not cfg.geometry.flip_latitude_to_north_south:
        raise ValueError(
            "geometry.flip_latitude_to_north_south must be true for "
            "legendre-gauss training"
        )
    if not cfg.geometry.roll_longitude_to_0_2pi:
        raise ValueError(
            "geometry.roll_longitude_to_0_2pi must be true for "
            "legendre-gauss training"
        )
    if cfg.training.channel_loss_weights is not None:
        weight_keys = set(cfg.training.channel_loss_weights)
        expected_keys = set(PHYSICAL_STATE_FIELDS)
        if weight_keys != expected_keys:
            raise ValueError(
                "training.channel_loss_weights must define exactly "
                f"{list(PHYSICAL_STATE_FIELDS)}, got {sorted(weight_keys)}"
            )
        for field_name in PHYSICAL_STATE_FIELDS:
            weight = float(cfg.training.channel_loss_weights[field_name])
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(
                    "training.channel_loss_weights values must be finite and > 0: "
                    f"{field_name}={weight!r}"
                )


def _validate_parameter_specs(specs: List[ParameterSpec]) -> None:
    """Validate sampling semantics and alias constraints."""
    names = [spec.name for spec in specs]
    dupes = sorted({name for name in names if names.count(name) > 1})
    if dupes:
        raise ValueError(f"sampling.parameters contains duplicate names: {dupes}")

    forbidden = sorted(set(names).intersection(INTERNAL_FIXED_PARAM_NAMES))
    if forbidden:
        raise ValueError("sampling.parameters must not include internal fixed diffusion parameters")

    allowed = set(USER_CORE_PARAM_NAMES).union(TAURAD_ALIASES).union(TAUDRAG_ALIASES)
    unknown = sorted(set(names).difference(allowed))
    if unknown:
        raise ValueError(f"sampling.parameters contains unknown names: {unknown}")

    missing_core = [name for name in USER_CORE_PARAM_NAMES if name not in names]
    if missing_core:
        raise ValueError(f"sampling.parameters missing required core names: {missing_core}")

    has_taurad_s = TAURAD_ALIASES[0] in names
    has_taurad_hours = TAURAD_ALIASES[1] in names
    if has_taurad_s == has_taurad_hours:
        raise ValueError(
            "sampling.parameters must include exactly one of "
            "['taurad_s','taurad_hours']"
        )

    has_taudrag_s = TAUDRAG_ALIASES[0] in names
    has_taudrag_hours = TAUDRAG_ALIASES[1] in names
    if has_taudrag_s == has_taudrag_hours:
        raise ValueError(
            "sampling.parameters must include exactly one of "
            "['taudrag_s','taudrag_hours']"
        )

    for spec in specs:
        if spec.dist not in VALID_PARAM_DISTS:
            raise ValueError(f"Unsupported dist '{spec.dist}' for parameter '{spec.name}'")
        if spec.dist in {"uniform", "loguniform"}:
            if spec.min is None or spec.max is None:
                raise ValueError(f"{spec.dist} requires min/max for '{spec.name}'")
            if not (_is_finite(spec.min) and _is_finite(spec.max)):
                raise ValueError(f"{spec.dist} min/max must be finite for '{spec.name}'")
            if float(spec.min) >= float(spec.max):
                raise ValueError(f"{spec.dist} requires min < max for '{spec.name}'")
            if spec.dist == "loguniform" and (float(spec.min) <= 0 or float(spec.max) <= 0):
                raise ValueError(f"loguniform requires positive min/max for '{spec.name}'")
            continue
        if spec.dist in {"const", "fixed"}:
            if spec.value is None:
                raise ValueError(f"{spec.dist} requires value for '{spec.name}'")
            if not _is_finite(spec.value):
                raise ValueError(f"{spec.dist} value must be finite for '{spec.name}'")
            continue
        if spec.dist == "mixture_off_loguniform":
            if (
                spec.p_off is None
                or spec.off_value is None
                or spec.on_min is None
                or spec.on_max is None
            ):
                raise ValueError(
                    "mixture_off_loguniform requires "
                    "p_off/off_value/on_min/on_max "
                    f"for '{spec.name}'"
                )
            if not _is_finite(spec.p_off) or not (
                PROBABILITY_MIN <= float(spec.p_off) <= PROBABILITY_MAX
            ):
                raise ValueError(
                    "mixture_off_loguniform p_off must be in [0,1] "
                    f"for '{spec.name}'"
                )
            if not (
                _is_finite(spec.on_min)
                and _is_finite(spec.on_max)
                and _is_finite(spec.off_value)
            ):
                raise ValueError(
                    "mixture_off_loguniform values must be finite "
                    f"for '{spec.name}'"
                )
            if float(spec.on_min) <= 0 or float(spec.on_max) <= 0:
                raise ValueError(
                    "mixture_off_loguniform on_min/on_max must be positive "
                    f"for '{spec.name}'"
                )
            if float(spec.on_min) >= float(spec.on_max):
                raise ValueError(
                    "mixture_off_loguniform requires on_min < on_max "
                    f"for '{spec.name}'"
                )


def _is_finite(value: float | int) -> bool:
    """Return True when the value is finite."""
    return math.isfinite(float(value))


def _validate_raw_config_keys(raw: Dict[str, Any]) -> None:
    """Reject unknown top-level and section keys before parsing values."""
    _reject_unknown_keys(raw, TOP_LEVEL_CONFIG_KEYS, "config")
    _validate_optional_section(raw, "paths", PATHS_KEYS)
    _validate_optional_section(raw, "solver", SOLVER_KEYS)
    _validate_optional_section(raw, "geometry", GEOMETRY_KEYS)
    _validate_sampling_keys(raw)
    _validate_normalization_keys(raw)
    _validate_model_keys(raw)
    _validate_training_keys(raw)


def _validate_optional_section(raw: Dict[str, Any], section: str, allowed_keys: set[str]) -> None:
    """Validate one optional top-level section against its allowed keys."""
    if section not in raw:
        return
    obj = raw[section]
    if not isinstance(obj, dict):
        raise ValueError(f"config.{section} must be an object")
    _reject_unknown_keys(obj, allowed_keys, f"config.{section}")


def _validate_sampling_keys(raw: Dict[str, Any]) -> None:
    """Validate key names for the ``sampling`` section."""
    if "sampling" not in raw:
        return
    sampling_obj = raw["sampling"]
    if not isinstance(sampling_obj, dict):
        raise ValueError("config.sampling must be an object")
    _reject_unknown_keys(sampling_obj, SAMPLING_KEYS, "config.sampling")
    params_obj = sampling_obj.get("parameters")
    if params_obj is None:
        return
    if not isinstance(params_obj, list):
        raise ValueError("config.sampling.parameters must be a list")
    for i, item in enumerate(params_obj):
        if not isinstance(item, dict):
            raise ValueError(f"config.sampling.parameters[{i}] must be an object")
        _reject_unknown_keys(item, PARAM_SPEC_KEYS, f"config.sampling.parameters[{i}]")


def _validate_normalization_keys(raw: Dict[str, Any]) -> None:
    """Validate key names for the ``normalization`` section."""
    if "normalization" not in raw:
        return
    norm_obj = raw["normalization"]
    if not isinstance(norm_obj, dict):
        raise ValueError("config.normalization must be an object")
    _reject_unknown_keys(norm_obj, NORMALIZATION_KEYS, "config.normalization")
    state_obj = norm_obj.get("state")
    if state_obj is not None:
        if not isinstance(state_obj, dict):
            raise ValueError("config.normalization.state must be an object")
        _reject_unknown_keys(state_obj, NORM_STATE_KEYS, "config.normalization.state")
        if "field_transforms" in state_obj and not isinstance(state_obj["field_transforms"], dict):
            raise ValueError("config.normalization.state.field_transforms must be an object")
    params_obj = norm_obj.get("params")
    if params_obj is not None:
        if not isinstance(params_obj, dict):
            raise ValueError("config.normalization.params must be an object")
        _reject_unknown_keys(params_obj, NORM_PARAMS_KEYS, "config.normalization.params")


def _validate_model_keys(raw: Dict[str, Any]) -> None:
    """Validate top-level model keys against the supported schema."""
    if "model" not in raw:
        return
    model_obj = raw["model"]
    if not isinstance(model_obj, dict):
        raise ValueError("config.model must be an object")
    _reject_unknown_keys(model_obj, MODEL_KEYS, "config.model")


def _validate_training_keys(raw: Dict[str, Any]) -> None:
    """Validate top-level training keys and nested scheduler keys."""
    if "training" not in raw:
        return
    train_obj = raw["training"]
    if not isinstance(train_obj, dict):
        raise ValueError("config.training must be an object")
    _reject_unknown_keys(train_obj, TRAINING_KEYS, "config.training")
    if "scheduler" in train_obj:
        scheduler_obj = train_obj["scheduler"]
        if not isinstance(scheduler_obj, dict):
            raise ValueError("config.training.scheduler must be an object")
        _reject_unknown_keys(scheduler_obj, SCHEDULER_KEYS, "config.training.scheduler")


def _reject_unknown_keys(obj: Dict[str, Any], allowed_keys: set[str], path: str) -> None:
    """Raise when a config object contains unsupported public keys."""
    unknown = sorted(k for k in obj.keys() if not str(k).startswith("_") and k not in allowed_keys)
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {unknown}")
