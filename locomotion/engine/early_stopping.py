"""Early stopping with best-weight restoration."""

from __future__ import annotations

import copy

import torch.nn as nn


class EarlyStopping:
    """Stop training when the monitored value stops improving.

    The paper trains every architecture with early stopping; this class also
    keeps a copy of the best weights so that evaluation never uses an
    over-fitted final epoch.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best_value: float | None = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
        self._best_state = None

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def step(self, value: float, model: nn.Module, epoch: int) -> bool:
        """Record ``value``; returns ``True`` when it is a new best."""
        if self._is_better(value):
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.counter = 0
            self._best_state = copy.deepcopy(model.state_dict())
            return True
        self.counter += 1
        if self.patience > 0 and self.counter >= self.patience:
            self.should_stop = True
        return False

    def restore(self, model: nn.Module) -> None:
        """Load the best weights seen so far back into ``model``."""
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
