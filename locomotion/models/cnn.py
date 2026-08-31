"""1-D convolutional baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseTimeSeriesModel, ClassificationHead, ConvBlock, safe_pool_sizes


class CNN(BaseTimeSeriesModel):
    """Stack of Conv1d + BatchNorm + ReLU + MaxPool blocks with a dense head.

    Global average pooling replaces the ``Flatten`` of the original Keras
    implementation, which makes the model independent of the window length.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        width: int = 128,
        depth: int = 3,
        kernel_size: int = 3,
        head_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__(c_in, c_out, seq_len)
        pools = safe_pool_sizes(seq_len, depth)
        blocks = []
        in_channels = c_in
        for pool in pools:
            blocks.append(ConvBlock(in_channels, width, kernel_size, pool))
            in_channels = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = ClassificationHead(width, head_dim, c_out, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)
