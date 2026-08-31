"""Mamba: selective state-space model for time series.

The original paper code relied on ``mamba_ssm``, which needs a CUDA build
toolchain. This module ships a pure-PyTorch selective scan so the model runs
anywhere (CPU included); when ``mamba_ssm`` is installed and a CUDA device is
available, its fused kernel is used instead for speed.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTimeSeriesModel, ClassificationHead

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - depends on the local CUDA toolchain
    from mamba_ssm import Mamba as _FusedMamba
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _FusedMamba = None


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation, as used in the Mamba paper."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)


class MambaBlock(nn.Module):
    """Single selective state-space block (S6), implemented in plain PyTorch."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        self.in_projection = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_projection = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_projection = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # A is parameterised in log space to keep it negative (stable dynamics).
        state_index = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(state_index).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_projection = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        """Sequential selective scan. ``u``: ``(B, L, d_inner)``."""
        batch, length, _ = u.shape
        A = -torch.exp(self.A_log)                                  # (d_inner, N)
        projected = self.x_projection(u)
        delta, B, C = torch.split(
            projected, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_projection(delta))               # (B, L, d_inner)

        delta_A = torch.exp(delta.unsqueeze(-1) * A)                # (B, L, d_inner, N)
        delta_Bu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        state = torch.zeros(
            batch, self.d_inner, self.d_state, device=u.device, dtype=u.dtype
        )
        outputs = []
        for step in range(length):
            state = delta_A[:, step] * state + delta_Bu[:, step]
            outputs.append(torch.einsum("bdn,bn->bd", state, C[:, step]))
        return torch.stack(outputs, dim=1) + u * self.D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        x_and_res = self.in_projection(x)
        u, res = x_and_res.chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[:, :, :length].transpose(1, 2)
        u = F.silu(u)
        y = self._selective_scan(u) * F.silu(res)
        return self.out_projection(y)


class _FusedMambaBlock(nn.Module):
    """Thin adapter around the CUDA kernel of ``mamba_ssm``."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.inner = _FusedMamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


class Mamba(BaseTimeSeriesModel):
    """Stack of residual Mamba blocks with mean pooling and a dense head."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        d_model: int = 128,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        head_dim: int = 128,
        dropout: float = 0.2,
        use_fused: bool = True,
    ):
        super().__init__(c_in, c_out, seq_len)
        fused = use_fused and _FusedMamba is not None and torch.cuda.is_available()
        if use_fused and not fused:
            LOGGER.info(
                "mamba_ssm CUDA kernel unavailable; using the portable PyTorch scan"
            )
        self.fused = fused

        self.input_projection = nn.Linear(c_in, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            block = (
                _FusedMambaBlock(d_model, d_state, d_conv, expand)
                if fused
                else MambaBlock(d_model, d_state, d_conv, expand)
            )
            self.layers.append(nn.ModuleDict({"norm": RMSNorm(d_model), "block": block}))
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = ClassificationHead(d_model, head_dim, c_out, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = self.input_projection(x.transpose(1, 2))   # (B, T, d_model)
        for layer in self.layers:
            x = x + layer["block"](layer["norm"](x))
        x = self.dropout(self.norm(x))
        return self.head(x.mean(dim=1))
