from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from .constants import MIN_ROLLOUT_STEPS, PROBABILITY_MAX, PROBABILITY_MIN


TransformName = Literal["none", "log10", "signed_log1p"]


@dataclass(frozen=True)
class Extended9Params:
    """Extended retrieval parameter set for terminal-state emulation."""

    a_m: float
    omega_rad_s: float
    Phibar: float
    DPhieq: float
    taurad_s: float
    taudrag_s: float
    g_m_s2: float
    K6: float
    K6Phi: Optional[float]

    def to_vector(self) -> np.ndarray:
        k6phi = 0.0 if self.K6Phi is None else float(self.K6Phi)
        return np.array(
            [
                self.a_m,
                self.omega_rad_s,
                self.Phibar,
                self.DPhieq,
                self.taurad_s,
                self.taudrag_s,
                self.g_m_s2,
                self.K6,
                k6phi,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class TerminalState:
    """Terminal shallow-water state in physical space."""

    phi: np.ndarray
    u: np.ndarray
    v: np.ndarray
    eta: np.ndarray
    delta: np.ndarray

    def as_stacked(self) -> np.ndarray:
        return np.stack([self.phi, self.u, self.v, self.eta, self.delta], axis=0)


@dataclass(frozen=True)
class NormalizationConfig:
    """Configurable per-channel normalization transforms before z-score."""

    field_transforms: Dict[str, TransformName] = field(
        default_factory=lambda: {
            "Phi": "none",
            "U": "none",
            "V": "none",
            "eta": "none",
            "delta": "none",
        }
    )
    zscore_eps: float = 1e-6
    log10_eps: float = 1e-30
    signed_log1p_scale: float = 1.0


@dataclass(frozen=True)
class PathsConfig:
    dataset_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    model_dir: str = "data/model"
    overwrite_dataset: bool = False
    overwrite_processed: bool = False


@dataclass(frozen=True)
class SolverConfig:
    M: int = 42
    dt_seconds: float = 60.0
    default_time_days: float = 100.0
    starttime_index: int = 2


@dataclass(frozen=True)
class GeometryConfig:
    flip_latitude_to_north_south: bool = True
    roll_longitude_to_0_2pi: bool = True


@dataclass(frozen=True)
class ParameterSpec:
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
    seed: int = 0
    n_sims: int = 500
    generation_workers: int = 0
    parameters: List[ParameterSpec] = field(default_factory=list)


@dataclass(frozen=True)
class ParamNormConfig:
    mode: str = "zscore"
    eps: float = 1e-6


@dataclass(frozen=True)
class NormalizationSettings:
    state: NormalizationConfig = field(default_factory=NormalizationConfig)
    params: ParamNormConfig = field(default_factory=ParamNormConfig)


@dataclass(frozen=True)
class ICConfig:
    basis: List[str] = field(default_factory=lambda: ["const", "sin_lat", "cos_lat", "sin_lon", "cos_lon"])
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "gelu"
    rand_basis_seed: int = 1234
    rand_basis_count: int = 2
    rand_basis_max_k: int = 3
    out_tanh: bool = True


@dataclass(frozen=True)
class ModelConfig:
    grid: str = "legendre-gauss"
    grid_internal: str = "legendre-gauss"
    scale_factor: int = 1
    embed_dim: int = 32
    num_layers: int = 4
    encoder_layers: int = 1
    activation_function: str = "gelu"
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    normalization_layer: str = "none"
    hard_thresholding_fraction: float = 1.0
    residual_prediction: bool = True
    pos_embed: str = "spectral"
    bias: bool = False
    include_param_maps: bool = True
    include_coord_channels: bool = True
    rollout_steps_at_default_time: int = 16
    ic: ICConfig = field(default_factory=ICConfig)


@dataclass(frozen=True)
class SchedulerConfig:
    type: str = "plateau"
    factor: float = 0.5
    patience: int = 10
    min_lr: float = 1e-6


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    device: str = "auto"
    amp_mode: str = "none"
    epochs: int = 50
    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    val_fraction: float = 0.2
    split_seed: int = 0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass(frozen=True)
class GCMulatorConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    normalization: NormalizationSettings = field(default_factory=NormalizationSettings)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


EXTENDED9_CORE_PARAM_NAMES = (
    "a_m",
    "omega_rad_s",
    "Phibar",
    "DPhieq",
    "g_m_s2",
    "K6",
    "K6Phi",
)
TAURAD_ALIASES = ("taurad_s", "taurad_hours")
TAUDRAG_ALIASES = ("taudrag_s", "taudrag_hours")
VALID_PARAM_DISTS = {"uniform", "loguniform", "const", "fixed", "mixture_off_loguniform"}
TOP_LEVEL_CONFIG_KEYS = {
    "paths",
    "solver",
    "geometry",
    "sampling",
    "normalization",
    "model",
    "training",
}
PATHS_KEYS = {"dataset_dir", "processed_dir", "model_dir", "overwrite_dataset", "overwrite_processed"}
SOLVER_KEYS = {"M", "dt_seconds", "default_time_days", "starttime_index"}
GEOMETRY_KEYS = {"flip_latitude_to_north_south", "roll_longitude_to_0_2pi"}
SAMPLING_KEYS = {"seed", "n_sims", "generation_workers", "parameters"}
PARAM_SPEC_KEYS = {"name", "dist", "min", "max", "value", "p_off", "off_value", "on_min", "on_max"}
NORMALIZATION_KEYS = {"state", "params"}
NORM_STATE_KEYS = {"field_transforms", "zscore_eps", "log10_eps", "signed_log1p_scale"}
NORM_FIELD_TRANSFORM_KEYS = {"Phi", "U", "V", "eta", "delta"}
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
    "pos_embed",
    "bias",
    "include_param_maps",
    "include_coord_channels",
    "rollout_steps_at_default_time",
    "ic",
}
MODEL_IC_KEYS = {
    "basis",
    "hidden_dim",
    "num_layers",
    "activation",
    "rand_basis_seed",
    "rand_basis_count",
    "rand_basis_max_k",
    "out_tanh",
}
TRAINING_KEYS = {
    "seed",
    "device",
    "amp_mode",
    "epochs",
    "batch_size",
    "num_workers",
    "shuffle",
    "pin_memory",
    "learning_rate",
    "weight_decay",
    "grad_clip_norm",
    "val_fraction",
    "split_seed",
    "scheduler",
}
SCHEDULER_KEYS = {"type", "factor", "patience", "min_lr"}


def _merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("YAML config requested but PyYAML is not installed") from exc
        obj = yaml.safe_load(text)
    else:
        obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("Config must be a JSON/YAML object")
    return obj


def _parse_paths(d: Dict[str, Any]) -> PathsConfig:
    return PathsConfig(
        dataset_dir=str(d.get("dataset_dir", "data/raw")),
        processed_dir=str(d.get("processed_dir", "data/processed")),
        model_dir=str(d.get("model_dir", "data/model")),
        overwrite_dataset=bool(d.get("overwrite_dataset", False)),
        overwrite_processed=bool(d.get("overwrite_processed", False)),
    )


def _parse_solver(d: Dict[str, Any]) -> SolverConfig:
    return SolverConfig(
        M=int(d.get("M", 42)),
        dt_seconds=float(d.get("dt_seconds", 60.0)),
        default_time_days=float(d.get("default_time_days", 100.0)),
        starttime_index=int(d.get("starttime_index", 2)),
    )


def _parse_geometry(d: Dict[str, Any]) -> GeometryConfig:
    return GeometryConfig(
        flip_latitude_to_north_south=bool(d.get("flip_latitude_to_north_south", True)),
        roll_longitude_to_0_2pi=bool(d.get("roll_longitude_to_0_2pi", True)),
    )


def _parse_parameter_specs(v: Any) -> List[ParameterSpec]:
    if not isinstance(v, list):
        raise ValueError("sampling.parameters must be a list")
    items = v
    out: List[ParameterSpec] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"sampling.parameters[{idx}] must be an object, got {type(it).__name__}")
        if "name" not in it:
            raise ValueError(f"sampling.parameters[{idx}] missing required key 'name'")
        if "dist" not in it:
            raise ValueError(f"sampling.parameters[{idx}] missing required key 'dist'")
        out.append(
            ParameterSpec(
                name=str(it["name"]),
                dist=str(it["dist"]),
                min=it.get("min"),
                max=it.get("max"),
                value=it.get("value"),
                p_off=it.get("p_off"),
                off_value=it.get("off_value"),
                on_min=it.get("on_min"),
                on_max=it.get("on_max"),
            )
        )
    return out


def _parse_sampling(d: Dict[str, Any]) -> SamplingConfig:
    specs = _parse_parameter_specs(d.get("parameters", []))
    if not specs:
        raise ValueError(
            "sampling.parameters is empty or missing in config. "
            "You must explicitly list all 9 Extended-9 parameters."
        )
    return SamplingConfig(
        seed=int(d.get("seed", 0)),
        n_sims=int(d.get("n_sims", 500)),
        generation_workers=int(d.get("generation_workers", 0)),
        parameters=specs,
    )


def _parse_norm(d: Dict[str, Any]) -> NormalizationSettings:
    state = d.get("state", {}) if isinstance(d.get("state", {}), dict) else {}
    params = d.get("params", {}) if isinstance(d.get("params", {}), dict) else {}

    field_transforms = state.get("field_transforms", {})
    if not isinstance(field_transforms, dict):
        field_transforms = {}

    state_cfg = NormalizationConfig(
        field_transforms={
            "Phi": str(field_transforms.get("Phi", "none")),
            "U": str(field_transforms.get("U", "none")),
            "V": str(field_transforms.get("V", "none")),
            "eta": str(field_transforms.get("eta", "none")),
            "delta": str(field_transforms.get("delta", "none")),
        },
        zscore_eps=float(state.get("zscore_eps", 1e-6)),
        log10_eps=float(state.get("log10_eps", 1e-30)),
        signed_log1p_scale=float(state.get("signed_log1p_scale", 1.0)),
    )

    params_cfg = ParamNormConfig(mode=str(params.get("mode", "zscore")), eps=float(params.get("eps", 1e-6)))

    return NormalizationSettings(state=state_cfg, params=params_cfg)


def _parse_ic(d: Dict[str, Any]) -> ICConfig:
    return ICConfig(
        basis=[str(x) for x in d.get("basis", ["const", "sin_lat", "cos_lat", "sin_lon", "cos_lon"])],
        hidden_dim=int(d.get("hidden_dim", 128)),
        num_layers=int(d.get("num_layers", 2)),
        activation=str(d.get("activation", "gelu")),
        rand_basis_seed=int(d.get("rand_basis_seed", 1234)),
        rand_basis_count=int(d.get("rand_basis_count", 2)),
        rand_basis_max_k=int(d.get("rand_basis_max_k", 3)),
        out_tanh=bool(d.get("out_tanh", True)),
    )


def _parse_model(d: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        grid=str(d.get("grid", "legendre-gauss")),
        grid_internal=str(d.get("grid_internal", "legendre-gauss")),
        scale_factor=int(d.get("scale_factor", 1)),
        embed_dim=int(d.get("embed_dim", 32)),
        num_layers=int(d.get("num_layers", 4)),
        encoder_layers=int(d.get("encoder_layers", 1)),
        activation_function=str(d.get("activation_function", "gelu")),
        use_mlp=bool(d.get("use_mlp", True)),
        mlp_ratio=float(d.get("mlp_ratio", 2.0)),
        drop_rate=float(d.get("drop_rate", 0.0)),
        drop_path_rate=float(d.get("drop_path_rate", 0.0)),
        normalization_layer=str(d.get("normalization_layer", "none")),
        hard_thresholding_fraction=float(d.get("hard_thresholding_fraction", 1.0)),
        residual_prediction=bool(d.get("residual_prediction", True)),
        pos_embed=str(d.get("pos_embed", "spectral")),
        bias=bool(d.get("bias", False)),
        include_param_maps=bool(d.get("include_param_maps", True)),
        include_coord_channels=bool(d.get("include_coord_channels", True)),
        rollout_steps_at_default_time=int(d.get("rollout_steps_at_default_time", 16)),
        ic=_parse_ic(d.get("ic", {}) if isinstance(d.get("ic", {}), dict) else {}),
    )


def _parse_scheduler(d: Dict[str, Any]) -> SchedulerConfig:
    return SchedulerConfig(
        type=str(d.get("type", "plateau")),
        factor=float(d.get("factor", 0.5)),
        patience=int(d.get("patience", 10)),
        min_lr=float(d.get("min_lr", 1e-6)),
    )


def _parse_training(d: Dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        seed=int(d.get("seed", 0)),
        device=str(d.get("device", "auto")),
        amp_mode=str(d.get("amp_mode", "none")),
        epochs=int(d.get("epochs", 50)),
        batch_size=int(d.get("batch_size", 8)),
        num_workers=int(d.get("num_workers", 0)),
        shuffle=bool(d.get("shuffle", True)),
        pin_memory=bool(d.get("pin_memory", False)),
        learning_rate=float(d.get("learning_rate", 3e-4)),
        weight_decay=float(d.get("weight_decay", 0.0)),
        grad_clip_norm=float(d.get("grad_clip_norm", 1.0)),
        val_fraction=float(d.get("val_fraction", 0.2)),
        split_seed=int(d.get("split_seed", 0)),
        scheduler=_parse_scheduler(d.get("scheduler", {}) if isinstance(d.get("scheduler", {}), dict) else {}),
    )


def load_config(config_path: Path) -> GCMulatorConfig:
    raw = _load_raw_config(config_path)
    _validate_raw_config_keys(raw)

    defaults = asdict(GCMulatorConfig())
    merged = _merge_dicts(defaults, raw)

    cfg = GCMulatorConfig(
        paths=_parse_paths(merged.get("paths", {})),
        solver=_parse_solver(merged.get("solver", {})),
        geometry=_parse_geometry(merged.get("geometry", {})),
        sampling=_parse_sampling(merged.get("sampling", {})),
        normalization=_parse_norm(merged.get("normalization", {})),
        model=_parse_model(merged.get("model", {})),
        training=_parse_training(merged.get("training", {})),
    )

    validate_config(cfg)
    return cfg


def resolve_path(config_path: Path, path_value: str) -> Path:
    return (config_path.parent / path_value).resolve()


def validate_config(cfg: GCMulatorConfig) -> None:
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
    _validate_parameter_specs(cfg.sampling.parameters)

    valid_transforms = {"none", "log10", "signed_log1p"}
    for field_name, tr in cfg.normalization.state.field_transforms.items():
        if tr not in valid_transforms:
            raise ValueError(f"Invalid transform for {field_name}: {tr}")
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

    if cfg.training.amp_mode not in {"none", "bf16", "fp16"}:
        raise ValueError("training.amp_mode must be one of ['none','bf16','fp16']")
    if cfg.training.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("training.device must be one of ['auto','cpu','cuda','mps']")
    if cfg.training.epochs < 1:
        raise ValueError("training.epochs must be >= 1")
    if cfg.training.batch_size < 1:
        raise ValueError("training.batch_size must be >= 1")
    if cfg.training.num_workers < 0:
        raise ValueError("training.num_workers must be >= 0")
    if cfg.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be > 0")
    if cfg.training.weight_decay < 0:
        raise ValueError("training.weight_decay must be >= 0")
    if cfg.training.grad_clip_norm < 0:
        raise ValueError("training.grad_clip_norm must be >= 0")
    if not (0.0 < cfg.training.val_fraction < 1.0):
        raise ValueError("training.val_fraction must be in (0,1)")
    if cfg.training.scheduler.type not in {"plateau", "none"}:
        raise ValueError("training.scheduler.type must be one of ['plateau','none']")
    if cfg.training.scheduler.type == "plateau":
        if cfg.training.scheduler.factor <= 0 or cfg.training.scheduler.factor >= 1:
            raise ValueError("training.scheduler.factor must be in (0,1)")
        if cfg.training.scheduler.patience < 0:
            raise ValueError("training.scheduler.patience must be >= 0")
        if cfg.training.scheduler.min_lr < 0:
            raise ValueError("training.scheduler.min_lr must be >= 0")

    if cfg.model.rollout_steps_at_default_time < MIN_ROLLOUT_STEPS:
        raise ValueError(
            f"model.rollout_steps_at_default_time must be >= {MIN_ROLLOUT_STEPS}"
        )
    if cfg.model.ic.hidden_dim < 1:
        raise ValueError("model.ic.hidden_dim must be >= 1")
    if cfg.model.ic.num_layers < 1:
        raise ValueError("model.ic.num_layers must be >= 1")
    if cfg.model.ic.activation not in {"relu", "gelu", "silu"}:
        raise ValueError("model.ic.activation must be one of ['relu','gelu','silu']")
    if cfg.model.ic.rand_basis_count < 0:
        raise ValueError("model.ic.rand_basis_count must be >= 0")
    if cfg.model.ic.rand_basis_max_k < 1:
        raise ValueError("model.ic.rand_basis_max_k must be >= 1")
    if not cfg.model.ic.basis and cfg.model.ic.rand_basis_count == 0:
        raise ValueError("model.ic.basis cannot be empty when rand_basis_count is 0")


def _validate_parameter_specs(specs: List[ParameterSpec]) -> None:
    names = [spec.name for spec in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"sampling.parameters contains duplicate names: {dupes}")

    allowed = set(EXTENDED9_CORE_PARAM_NAMES).union(TAURAD_ALIASES).union(TAUDRAG_ALIASES)
    unknown = sorted(set(names).difference(allowed))
    if unknown:
        raise ValueError(f"sampling.parameters contains unknown names: {unknown}")

    missing_core = [name for name in EXTENDED9_CORE_PARAM_NAMES if name not in names]
    if missing_core:
        raise ValueError(f"sampling.parameters missing required Extended-9 names: {missing_core}")

    has_taurad_s = TAURAD_ALIASES[0] in names
    has_taurad_hours = TAURAD_ALIASES[1] in names
    if has_taurad_s == has_taurad_hours:
        raise ValueError("sampling.parameters must include exactly one of ['taurad_s','taurad_hours']")

    has_taudrag_s = TAUDRAG_ALIASES[0] in names
    has_taudrag_hours = TAUDRAG_ALIASES[1] in names
    if has_taudrag_s == has_taudrag_hours:
        raise ValueError("sampling.parameters must include exactly one of ['taudrag_s','taudrag_hours']")

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
            if spec.p_off is None or spec.off_value is None or spec.on_min is None or spec.on_max is None:
                raise ValueError(f"mixture_off_loguniform requires p_off/off_value/on_min/on_max for '{spec.name}'")
            if not (_is_finite(spec.p_off) and PROBABILITY_MIN <= float(spec.p_off) <= PROBABILITY_MAX):
                raise ValueError(f"mixture_off_loguniform p_off must be in [0,1] for '{spec.name}'")
            if not (_is_finite(spec.off_value) and _is_finite(spec.on_min) and _is_finite(spec.on_max)):
                raise ValueError(f"mixture_off_loguniform values must be finite for '{spec.name}'")
            if float(spec.on_min) <= 0 or float(spec.on_max) <= 0:
                raise ValueError(f"mixture_off_loguniform on_min/on_max must be positive for '{spec.name}'")
            if float(spec.on_min) >= float(spec.on_max):
                raise ValueError(f"mixture_off_loguniform requires on_min < on_max for '{spec.name}'")


def _is_finite(v: float | int) -> bool:
    return math.isfinite(float(v))


def _require_finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value}")
    return v


def time_days_to_rollout_steps(
    time_days: float,
    *,
    default_time_days: float,
    rollout_steps_at_default_time: int,
) -> int:
    days = _require_finite("time_days", time_days)
    default_days = _require_finite("default_time_days", default_time_days)
    if days <= 0:
        raise ValueError(f"time_days must be > 0, got {time_days}")
    if default_days <= 0:
        raise ValueError(f"default_time_days must be > 0, got {default_time_days}")

    steps_default = int(rollout_steps_at_default_time)
    if float(steps_default) != float(rollout_steps_at_default_time):
        raise ValueError(
            "rollout_steps_at_default_time must be an integer-like value, "
            f"got {rollout_steps_at_default_time}"
        )
    if steps_default < MIN_ROLLOUT_STEPS:
        raise ValueError(
            "rollout_steps_at_default_time must be >= "
            f"{MIN_ROLLOUT_STEPS}, got {rollout_steps_at_default_time}"
        )

    raw_steps = days / default_days * float(steps_default)
    if not math.isfinite(raw_steps):
        raise ValueError(
            "Computed rollout step ratio is non-finite "
            f"(time_days={time_days}, default_time_days={default_time_days}, "
            f"rollout_steps_at_default_time={rollout_steps_at_default_time})"
        )

    steps = int(round(raw_steps))
    if steps < MIN_ROLLOUT_STEPS:
        raise ValueError(
            f"Computed rollout steps={steps} (time_days={time_days}, "
            f"default_time_days={default_time_days}, "
            f"rollout_steps_at_default_time={rollout_steps_at_default_time}). "
            f"Must be >= {MIN_ROLLOUT_STEPS}."
        )
    return steps


def _validate_raw_config_keys(raw: Dict[str, Any]) -> None:
    _reject_unknown_keys(raw, TOP_LEVEL_CONFIG_KEYS, "config")

    _validate_optional_section(raw, "paths", PATHS_KEYS)
    _validate_optional_section(raw, "solver", SOLVER_KEYS)
    _validate_optional_section(raw, "geometry", GEOMETRY_KEYS)
    _validate_sampling_keys(raw)
    _validate_normalization_keys(raw)
    _validate_model_keys(raw)
    _validate_training_keys(raw)


def _validate_optional_section(raw: Dict[str, Any], section: str, allowed_keys: set[str]) -> None:
    if section not in raw:
        return
    section_obj = raw[section]
    if not isinstance(section_obj, dict):
        raise ValueError(f"config.{section} must be an object")
    _reject_unknown_keys(section_obj, allowed_keys, f"config.{section}")


def _validate_sampling_keys(raw: Dict[str, Any]) -> None:
    if "sampling" not in raw:
        return
    sampling_obj = raw["sampling"]
    if not isinstance(sampling_obj, dict):
        raise ValueError("config.sampling must be an object")
    _reject_unknown_keys(sampling_obj, SAMPLING_KEYS, "config.sampling")
    if "parameters" in sampling_obj:
        params_obj = sampling_obj["parameters"]
        if not isinstance(params_obj, list):
            raise ValueError("config.sampling.parameters must be a list")
        for i, item in enumerate(params_obj):
            if not isinstance(item, dict):
                raise ValueError(f"config.sampling.parameters[{i}] must be an object")
            _reject_unknown_keys(item, PARAM_SPEC_KEYS, f"config.sampling.parameters[{i}]")


def _validate_normalization_keys(raw: Dict[str, Any]) -> None:
    if "normalization" not in raw:
        return
    norm_obj = raw["normalization"]
    if not isinstance(norm_obj, dict):
        raise ValueError("config.normalization must be an object")
    _reject_unknown_keys(norm_obj, NORMALIZATION_KEYS, "config.normalization")
    if "state" in norm_obj:
        state_obj = norm_obj["state"]
        if not isinstance(state_obj, dict):
            raise ValueError("config.normalization.state must be an object")
        _reject_unknown_keys(state_obj, NORM_STATE_KEYS, "config.normalization.state")
        if "field_transforms" in state_obj:
            ft_obj = state_obj["field_transforms"]
            if not isinstance(ft_obj, dict):
                raise ValueError("config.normalization.state.field_transforms must be an object")
            _reject_unknown_keys(ft_obj, NORM_FIELD_TRANSFORM_KEYS, "config.normalization.state.field_transforms")
    if "params" in norm_obj:
        params_obj = norm_obj["params"]
        if not isinstance(params_obj, dict):
            raise ValueError("config.normalization.params must be an object")
        _reject_unknown_keys(params_obj, NORM_PARAMS_KEYS, "config.normalization.params")


def _validate_model_keys(raw: Dict[str, Any]) -> None:
    if "model" not in raw:
        return
    model_obj = raw["model"]
    if not isinstance(model_obj, dict):
        raise ValueError("config.model must be an object")
    _reject_unknown_keys(model_obj, MODEL_KEYS, "config.model")
    if "ic" in model_obj:
        ic_obj = model_obj["ic"]
        if not isinstance(ic_obj, dict):
            raise ValueError("config.model.ic must be an object")
        _reject_unknown_keys(ic_obj, MODEL_IC_KEYS, "config.model.ic")


def _validate_training_keys(raw: Dict[str, Any]) -> None:
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
    unknown = sorted(k for k in obj.keys() if not str(k).startswith("_") and k not in allowed_keys)
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {unknown}")


def dump_config(cfg: GCMulatorConfig, path: Path) -> None:
    d = asdict(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")
