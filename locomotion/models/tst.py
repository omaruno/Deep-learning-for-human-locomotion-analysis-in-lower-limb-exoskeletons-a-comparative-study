"""Time Series Transformer (TST).

Encoder-only transformer over the time axis, with a learnable positional
encoding and mean pooling over time -- the pooled variant keeps the model
usable at any ``--window-size`` without changing the head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseTimeSeriesModel, ClassificationHead


class TST(BaseTimeSeriesModel):
    """Transformer encoder for multivariate time series classification."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int | None = None,
        head_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__(c_in, c_out, seq_len)
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}); "
                "adjust --d-model or --n-heads"
            )
        self.input_projection = nn.Linear(c_in, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff or 4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = ClassificationHead(d_model, head_dim, c_out, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = x.transpose(1, 2)                       # (B, T, C)
        x = self.input_projection(x)
        length = x.shape[1]
        if length > self.positional_encoding.shape[1]:
            raise ValueError(
                f"TST was built for windows of at most {self.positional_encoding.shape[1]} "
                f"samples but received {length}"
            )
        x = self.dropout(x + self.positional_encoding[:, :length])
        x = self.norm(self.encoder(x))
        return self.head(x.mean(dim=1))             # mean pooling over time
