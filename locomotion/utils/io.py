"""JSON / YAML serialisation that tolerates NumPy types."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _to_builtin(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays and Paths to builtins."""
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None  # JSON has no NaN/Infinity literal
    return value


def save_json(data: Any, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_builtin(data), handle, indent=2, sort_keys=False)
    return path


def save_yaml(data: Any, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_builtin(data), handle, sort_keys=False, default_flow_style=False)
    return path


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
