"""Hybrid convolutional / recurrent models: CNN-LSTM and LSTM-CNN."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseTimeSeriesModel, ClassificationHead, ConvBlock, safe_pool_sizes


class CNNLSTM(BaseTimeSeriesModel):
    """Convolutional feature extractor followed by an LSTM.

    The convolutions compress the window into a shorter sequence of local
    motion features, which the LSTM then integrates over time.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        width: int = 128,
        depth: int = 3,
        kernel_size: int = 3,
        hidden: int = 128,
        n_layers: int = 1,
        head_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__(c_in, c_out, seq_len)
        blocks = []
        in_channels = c_in
        for pool in safe_pool_sizes(seq_len, depth):
            blocks.append(ConvBlock(in_channels, width, kernel_size, pool))
            in_channels = width
        self.features = nn.Sequential(*blocks)
        self.rnn = nn.LSTM(
            input_size=width,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = ClassificationHead(hidden, head_dim, c_out, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = self.features(x)               # (B, width, T')
        output, _ = self.rnn(x.transpose(1, 2))
        return self.head(output[:, -1])


class LSTMCNN(BaseTimeSeriesModel):
    """LSTM sequence encoder followed by convolutional blocks.

    The LSTM produces a context-aware representation at every time step, and
    the convolutions then look for discriminative patterns inside it.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        hidden: int = 128,
        n_layers: int = 1,
        width: int = 128,
        depth: int = 3,
        kernel_size: int = 3,
        head_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__(c_in, c_out, seq_len)
        self.rnn = nn.LSTM(
            input_size=c_in,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        blocks = []
        in_channels = hidden
        for pool in safe_pool_sizes(seq_len, depth):
            blocks.append(ConvBlock(in_channels, width, kernel_size, pool))
            in_channels = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = ClassificationHead(width, head_dim, c_out, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        output, _ = self.rnn(x.transpose(1, 2))   # (B, T, hidden)
        x = self.features(output.transpose(1, 2))
        x = self.pool(x).squeeze(-1)
        return self.head(x)
