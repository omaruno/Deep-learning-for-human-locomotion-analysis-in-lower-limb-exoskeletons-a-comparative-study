"""XceptionTime: depthwise-separable inception blocks for time series.

Re-implementation of Rahimian et al. (2019), following the reference
``tsai`` architecture but without the external dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseTimeSeriesModel


class SeparableConv1d(nn.Module):
    """Depthwise convolution followed by a pointwise (1x1) convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class XceptionModule(nn.Module):
    """Multi-scale separable convolutions plus a max-pool branch."""

    def __init__(self, in_channels: int, n_filters: int, kernel_size: int = 39):
        super().__init__()
        self.bottleneck = (
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
            if in_channels > 1
            else nn.Identity()
        )
        bottleneck_out = n_filters if in_channels > 1 else in_channels
        # Odd kernels keep the output length identical across the branches.
        kernel_sizes = [max(3, (kernel_size // (2**i)) // 2 * 2 + 1) for i in range(3)]
        self.convs = nn.ModuleList(
            SeparableConv1d(bottleneck_out, n_filters, ks) for ks in kernel_sizes
        )
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )
        self.norm = nn.BatchNorm1d(n_filters * 4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        branches = [conv(bottleneck) for conv in self.convs]
        branches.append(self.maxconvpool(x))
        return self.act(self.norm(torch.cat(branches, dim=1)))


class XceptionBlock(nn.Module):
    """Four Xception modules with a residual connection every two modules."""

    def __init__(self, in_channels: int, n_filters: int, n_modules: int = 4):
        super().__init__()
        self.n_modules = n_modules
        self.modules_list = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        channels = in_channels
        residual_channels = in_channels
        for index in range(n_modules):
            filters = n_filters * 2**index
            self.modules_list.append(XceptionModule(channels, filters))
            channels = filters * 4
            if index % 2 == 1:
                self.shortcuts.append(
                    nn.Sequential(
                        nn.Conv1d(residual_channels, channels, kernel_size=1, bias=False),
                        nn.BatchNorm1d(channels),
                    )
                )
                residual_channels = channels
        self.out_channels = channels
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        shortcut_index = 0
        for index, module in enumerate(self.modules_list):
            x = module(x)
            if index % 2 == 1:
                x = self.act(x + self.shortcuts[shortcut_index](residual))
                shortcut_index += 1
                residual = x
        return x


class XceptionTime(BaseTimeSeriesModel):
    """XceptionTime backbone with a fully convolutional head."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        width: int = 16,
        n_modules: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__(c_in, c_out, seq_len)
        self.block = XceptionBlock(c_in, width, n_modules)
        channels = self.block.out_channels
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 2, channels // 4, kernel_size=1),
            nn.BatchNorm1d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels // 4, c_out, kernel_size=1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return self.head(self.block(x))
