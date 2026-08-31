"""Seeding and device selection."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch.

    ``deterministic=True`` also pins the cuDNN algorithms, which makes runs
    bit-reproducible at some cost in speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    """Turn ``auto``/``cuda``/``cpu``/``mps`` into a concrete device."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)
