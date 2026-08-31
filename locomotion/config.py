"""The single configuration object shared by the CLI, the runner and the report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional

from .data.constants import DEFAULT_SAMPLING_RATE_HZ


@dataclass
class ExperimentConfig:
    """Everything one run needs. Serialised next to the results as ``config.yaml``."""

    # ---------------------------------------------------------------- data --- #
    data_dir: str = "data/processed"
    modality: str = "imu"                      # imu | emg | multimodal
    task: str = "classification"               # classification | ramp_slope | stair_height
    window_size: int = 100                     # samples per window
    window_ms: Optional[float] = None          # alternative to window_size, in ms
    stride: Optional[int] = None               # defaults to window_size (no overlap)
    sampling_rate: int = DEFAULT_SAMPLING_RATE_HZ
    imu_locations: Optional[List[str]] = None  # subset of foot/shank/thigh/trunk
    emg_muscles: Optional[List[str]] = None
    data_layout: str = "auto"
    file_pattern: str = "*"

    # --------------------------------------------------------------- model --- #
    model: str = "lstm"
    model_size: str = "base"                   # tiny | small | base | large
    width: Optional[int] = None                # per-field overrides of the preset
    depth: Optional[int] = None
    hidden: Optional[int] = None
    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    head_dim: Optional[int] = None
    dropout: float = 0.2

    # ------------------------------------------------------------ training --- #
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    grad_clip: float = 1.0
    val_fraction: float = 0.2
    class_weights: bool = False
    num_workers: int = 0

    # ---------------------------------------------------------- validation --- #
    evaluation: str = "loso"                   # loso | holdout
    subjects: Optional[List[str]] = None       # restrict the LOSO folds
    max_folds: Optional[int] = None
    test_fraction: float = 0.2                 # only used by 'holdout'

    # ---------------------------------------------------------------- shap --- #
    shap: bool = False
    shap_background: int = 100
    shap_samples: int = 200
    shap_folds: int = 1                        # explain the first N folds

    # -------------------------------------------------------------- runtime --- #
    device: str = "auto"
    seed: int = 42
    deterministic: bool = False
    output_dir: str = "outputs"
    run_name: Optional[str] = None
    cache_dir: Optional[str] = "outputs/cache"
    save_checkpoints: bool = True
    measure_latency: bool = True
    log_level: str = "INFO"

    # ------------------------------------------------------------- helpers --- #
    def __post_init__(self) -> None:
        # --window-ms is the paper's unit; convert it to samples once, here.
        if self.window_ms is not None:
            self.window_size = max(2, int(round(self.window_ms * self.sampling_rate / 1000)))
        if self.run_name is None:
            self.run_name = self.default_run_name()

    def default_run_name(self) -> str:
        parts = [self.model, self.modality, f"w{self.window_size}", self.model_size]
        if self.task != "classification":
            parts.insert(1, self.task)
        return "_".join(parts)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / str(self.run_name)

    @property
    def window_duration_ms(self) -> float:
        return 1000.0 * self.window_size / self.sampling_rate

    def size_overrides(self) -> dict:
        keys = ("width", "depth", "hidden", "d_model", "n_layers", "n_heads", "head_dim")
        return {key: getattr(self, key) for key in keys if getattr(self, key) is not None}

    def to_dict(self) -> dict:
        data = asdict(self)
        data["window_duration_ms"] = self.window_duration_ms
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        known = {f.name for f in fields(cls)}
        unknown = [key for key in data if key not in known]
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {unknown}")
        return cls(**{key: value for key, value in data.items() if key in known})

    @classmethod
    def from_yaml(cls, path) -> "ExperimentConfig":
        from .utils.io import load_yaml

        return cls.from_dict(load_yaml(path))
