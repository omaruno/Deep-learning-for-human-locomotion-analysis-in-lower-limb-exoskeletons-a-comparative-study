"""Torch dataset and per-channel standardisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ChannelScaler:
    """Per-channel z-score standardisation.

    Statistics are always fitted on the training split only, so that no
    information from the held-out subject leaks into the model.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-8) -> "ChannelScaler":
        # x: (n_windows, n_channels, window_size)
        mean = x.mean(axis=(0, 2), keepdims=True)
        std = np.maximum(x.std(axis=(0, 2), keepdims=True), eps)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mean": np.squeeze(self.mean).tolist(),
            "std": np.squeeze(self.std).tolist(),
        }


class WindowDataset(Dataset):
    """Windows as ``(channels, time)`` tensors plus their target."""

    def __init__(self, x: np.ndarray, y: np.ndarray, regression: bool = False):
        if len(x) != len(y):
            raise ValueError(f"x and y length mismatch: {len(x)} vs {len(y)}")
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        if regression:
            self.y = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1)
        else:
            self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.regression = regression

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    @property
    def n_channels(self) -> int:
        return int(self.x.shape[1])

    @property
    def window_size(self) -> int:
        return int(self.x.shape[2])
