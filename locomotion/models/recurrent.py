"""Recurrent baselines: LSTM and GRU."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseTimeSeriesModel, ClassificationHead


class _RecurrentNet(BaseTimeSeriesModel):
    """Shared implementation for the LSTM and GRU models."""

    rnn_class: type = nn.LSTM

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        hidden: int = 128,
        n_layers: int = 2,
        bidirectional: bool = False,
        head_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__(c_in, c_out, seq_len)
        self.rnn = self.rnn_class(
            input_size=c_in,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        n_features = hidden * (2 if bidirectional else 1)
        self.head = ClassificationHead(n_features, head_dim, c_out, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        # (B, C, T) -> (B, T, C): torch RNNs are time-major on dim 1.
        output, _ = self.rnn(x.transpose(1, 2))
        return self.head(output[:, -1])


class LSTM(_RecurrentNet):
    """Two-layer LSTM followed by a dense head (best model of the paper)."""

    rnn_class = nn.LSTM


class GRU(_RecurrentNet):
    """GRU counterpart of :class:`LSTM`."""

    rnn_class = nn.GRU
