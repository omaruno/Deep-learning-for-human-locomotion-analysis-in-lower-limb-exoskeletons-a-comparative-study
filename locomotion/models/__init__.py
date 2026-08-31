"""Model zoo: the eight architectures compared in the paper.

All of them share one interface -- ``(batch, channels, time)`` in,
``(batch, c_out)`` out -- and one capacity knob, ``--model-size``.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch.nn as nn

from .base import DEFAULT_SIZE, MODEL_SIZES, BaseTimeSeriesModel, SizeSpec, get_size_spec
from .cnn import CNN
from .hybrid import CNNLSTM, LSTMCNN
from .mamba import Mamba
from .recurrent import GRU, LSTM
from .tst import TST
from .xceptiontime import XceptionTime


def _build_cnn(c_in, c_out, seq_len, spec: SizeSpec, dropout: float) -> nn.Module:
    return CNN(
        c_in, c_out, seq_len,
        width=spec.width, depth=spec.depth, head_dim=spec.head_dim, dropout=dropout,
    )


def _build_lstm(c_in, c_out, seq_len, spec: SizeSpec, dropout: float) -> nn.Module:
    return LSTM(
        c_in, c_out, seq_len,
        hidden=spec.hidden, n_layers=max(2, spec.n_layers - 1),
        head_dim=spec.head_dim, dropout=dropout,
    )


def _build_gru(c_in, c_out, seq_len, spec: SizeSpec, dropout: float) -> nn.Module:
    return GRU(
        c_in, c_out, seq_len,
        hidden=spec.hidden, n_layers=max(1, spec.n_layers - 1),
        head_dim=spec.head_dim, dropout=dropout,
    )


def _build_cnn_lstm(c_in, c_out, seq_len, spec: SizeSpec, dropout: float) -> nn.Module:
    return CNNLSTM(
        c_in, c_out, seq_len,
        width=spec.width, depth=spec.depth, hidden=spec.hidden,
        head_dim=spec.head_dim, dropout=dropout,
    )


def _build_lstm_cnn(c_in, c_out, seq_len, spec: SizeSpec, dropout: float) -> nn.Module:
    return LSTMCNN(
        c_in, c_out, seq_len,
        hidden=spec.hidden, width=spec.width, depth=spec.depth,
        head_dim=spec.head_dim, dropout=dropout,
    )


def _build_xception(c_in, c_out, seq_len, spec: SizeSpec, dropout: float) -> nn.Module:
    # XceptionTime doubles its filters at every module, so it starts narrower.
    return XceptionTime(
        c_in, c_out, seq_len, width=max(8, spec.width // 8), dropout=dropout
    )


def _build_tst(c_in, c_out, seq_len, spec: SizeSpec, dropout: float) -> nn.Module:
    return TST(
        c_in, c_out, seq_len,
        d_model=spec.d_model, n_layers=spec.n_layers, n_heads=spec.n_heads,
        head_dim=spec.head_dim, dropout=dropout,
    )


def _build_mamba(c_in, c_out, seq_len, spec: SizeSpec, dropout: float) -> nn.Module:
    return Mamba(
        c_in, c_out, seq_len,
        d_model=spec.d_model, n_layers=spec.n_layers,
        head_dim=spec.head_dim, dropout=dropout,
    )


#: ``--model`` name -> builder. Keys are what the CLI accepts.
MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "cnn": _build_cnn,
    "lstm": _build_lstm,
    "gru": _build_gru,
    "cnn_lstm": _build_cnn_lstm,
    "lstm_cnn": _build_lstm_cnn,
    "xceptiontime": _build_xception,
    "tst": _build_tst,
    "mamba": _build_mamba,
}

#: Human-readable description, printed by ``--list-models``.
MODEL_DESCRIPTIONS: Dict[str, str] = {
    "cnn": "1-D convolutional network",
    "lstm": "Long short-term memory network (best terrain / ramp model in the paper)",
    "gru": "Gated recurrent unit network",
    "cnn_lstm": "Convolutional encoder followed by an LSTM (best stair-height model)",
    "lstm_cnn": "LSTM encoder followed by convolutional blocks",
    "xceptiontime": "XceptionTime, depthwise-separable inception blocks",
    "tst": "Time Series Transformer, attention-based encoder",
    "mamba": "Mamba selective state-space model",
}


def available_models() -> List[str]:
    return list(MODEL_REGISTRY)


def build_model(
    name: str,
    c_in: int,
    c_out: int,
    seq_len: int,
    size: str = DEFAULT_SIZE,
    dropout: float = 0.2,
    **size_overrides,
) -> nn.Module:
    """Instantiate a model by name.

    Args:
        name: key of :data:`MODEL_REGISTRY` (case-insensitive, ``-`` and ``_``
            are interchangeable).
        c_in: number of input channels, i.e. the selected modality's width.
        c_out: 5 for terrain classification, 1 for the regression tasks.
        seq_len: window length in samples.
        size: one of :data:`locomotion.models.base.MODEL_SIZES`.
        dropout: dropout probability used throughout the network.
        **size_overrides: per-field overrides of the size preset
            (``width``, ``depth``, ``hidden``, ``d_model``, ``n_layers``,
            ``n_heads``, ``head_dim``).
    """
    key = name.strip().lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'; available: {available_models()}"
        )
    spec = get_size_spec(size, **size_overrides)
    return MODEL_REGISTRY[key](c_in, c_out, seq_len, spec, dropout)


__all__ = [
    "BaseTimeSeriesModel",
    "CNN",
    "CNNLSTM",
    "DEFAULT_SIZE",
    "GRU",
    "LSTM",
    "LSTMCNN",
    "MODEL_DESCRIPTIONS",
    "MODEL_REGISTRY",
    "MODEL_SIZES",
    "Mamba",
    "SizeSpec",
    "TST",
    "XceptionTime",
    "available_models",
    "build_model",
    "get_size_spec",
]
