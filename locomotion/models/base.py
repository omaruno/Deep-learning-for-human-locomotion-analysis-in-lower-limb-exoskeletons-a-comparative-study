"""Shared building blocks and the ``--model-size`` presets.

Every model in this package consumes ``(batch, channels, time)`` tensors and
returns ``(batch, c_out)``: class logits for terrain classification, or a
single value for the regression tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SizeSpec:
    """Capacity preset shared by every architecture.

    Each model reads the fields that are meaningful to it: convolutional nets
    use ``width``/``depth``, recurrent nets use ``hidden``/``n_layers``,
    attention and state-space models use ``d_model``/``n_layers``/``n_heads``.

    Attributes:
        width: number of convolutional filters per block.
        depth: number of convolutional blocks.
        hidden: recurrent hidden size.
        d_model: embedding size of TST / Mamba.
        n_layers: number of recurrent, transformer or Mamba layers.
        n_heads: attention heads of the transformer.
        head_dim: width of the fully connected classification head.
    """

    width: int
    depth: int
    hidden: int
    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


#: ``--model-size`` presets. ``base`` reproduces the configuration of the paper
#: (128 filters / 128 hidden units / 128-unit dense head).
MODEL_SIZES: Dict[str, SizeSpec] = {
    "tiny": SizeSpec(width=32, depth=2, hidden=32, d_model=32, n_layers=1, n_heads=2, head_dim=32),
    "small": SizeSpec(width=64, depth=3, hidden=64, d_model=64, n_layers=2, n_heads=4, head_dim=64),
    "base": SizeSpec(width=128, depth=3, hidden=128, d_model=128, n_layers=3, n_heads=8, head_dim=128),
    "large": SizeSpec(width=256, depth=4, hidden=256, d_model=256, n_layers=4, n_heads=8, head_dim=256),
}
DEFAULT_SIZE = "base"


def get_size_spec(name: str, **overrides) -> SizeSpec:
    """Look up a preset and apply explicit command-line overrides."""
    if name not in MODEL_SIZES:
        raise ValueError(
            f"Unknown model size '{name}'; available: {list(MODEL_SIZES)}"
        )
    spec = MODEL_SIZES[name]
    clean = {key: value for key, value in overrides.items() if value is not None}
    unknown = [key for key in clean if key not in spec.to_dict()]
    if unknown:
        raise ValueError(f"Unknown size override(s) {unknown}")
    return SizeSpec(**{**spec.to_dict(), **clean})


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU, optionally followed by max pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.conv(x)))
        return self.drop(self.pool(x))


class ClassificationHead(nn.Module):
    """Dense head shared by every architecture."""

    def __init__(self, in_features: int, hidden: int, c_out: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseTimeSeriesModel(nn.Module):
    """Common interface: ``(B, C, T)`` in, ``(B, c_out)`` out."""

    def __init__(self, c_in: int, c_out: int, seq_len: int):
        super().__init__()
        self.c_in = int(c_in)
        self.c_out = int(c_out)
        self.seq_len = int(seq_len)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(
                f"{type(self).__name__} expects (batch, channels, time), got {tuple(x.shape)}"
            )


def safe_pool_sizes(seq_len: int, depth: int, pool: int = 2) -> list:
    """Pooling factor per block, disabled once the sequence gets too short.

    Stacking five ``MaxPool1d(2)`` layers on a 100-sample window (as the
    original notebooks did) collapses the time axis; this keeps deep configs
    usable on short windows.
    """
    sizes = []
    length = seq_len
    for _ in range(depth):
        if length // pool >= 4:
            sizes.append(pool)
            length //= pool
        else:
            sizes.append(1)
    return sizes
