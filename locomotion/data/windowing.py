"""Sliding-window segmentation of continuous recordings.

A window is kept only when every one of its samples carries the same activity
label (windows straddling a terrain transition are discarded), it never spans
two recordings, and it contains no missing values. This reproduces the
segmentation rule of the paper while allowing overlap through ``stride``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from .constants import (
    CLASS_TO_INDEX,
    LABEL_COLUMN,
    RAMP_TARGET_COLUMN,
    RECORDING_COLUMN,
    STAIR_TARGET_COLUMN,
    SUBJECT_COLUMN,
    normalise_label,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class WindowedData:
    """Windowed dataset held in memory.

    Attributes:
        x: ``(n_windows, n_channels, window_size)`` float32 array.
        y: ``(n_windows,)`` int64 terrain class indices.
        subjects: ``(n_windows,)`` subject identifier per window.
        ramp_incline: ``(n_windows,)`` ramp inclination in degrees (NaN if n/a).
        stair_height: ``(n_windows,)`` stair height in millimetres (NaN if n/a).
        channels: names of the channels, in array order.
        window_size: number of samples per window.
    """

    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    ramp_incline: np.ndarray
    stair_height: np.ndarray
    channels: List[str]
    window_size: int

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.x.shape[1])

    def subject_ids(self) -> List[str]:
        return sorted(set(self.subjects.tolist()))

    def select(self, index: np.ndarray) -> "WindowedData":
        return WindowedData(
            x=self.x[index],
            y=self.y[index],
            subjects=self.subjects[index],
            ramp_incline=self.ramp_incline[index],
            stair_height=self.stair_height[index],
            channels=list(self.channels),
            window_size=self.window_size,
        )


def _nanmean(values: np.ndarray) -> float:
    """Mean of the finite entries, or NaN when there are none.

    Regression targets are only defined for the relevant terrain (a ramp
    inclination is NaN on stairs), so all-NaN slices are the normal case here
    rather than an anomaly worth warning about.
    """
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def segment_dataframe(
    frame: pd.DataFrame,
    channels: Sequence[str],
    window_size: int,
    stride: int | None = None,
) -> WindowedData:
    """Cut ``frame`` into windows of ``window_size`` samples.

    Args:
        frame: concatenated recordings (see :func:`locomotion.data.io.load_recordings`).
        channels: channel columns to keep, in the desired order.
        window_size: samples per window.
        stride: hop between consecutive windows; defaults to ``window_size``
            (non-overlapping windows, as in the paper).
    """
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2, got {window_size}")
    stride = int(stride or window_size)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    channels = list(channels)
    x_windows: List[np.ndarray] = []
    y_windows: List[int] = []
    subject_windows: List[str] = []
    ramp_windows: List[float] = []
    stair_windows: List[float] = []
    n_mixed = 0
    n_idle = 0
    n_nan = 0

    grouping = [RECORDING_COLUMN, SUBJECT_COLUMN]
    for (_, subject), group in frame.groupby(grouping, sort=False):
        values = group[channels].to_numpy(dtype=np.float32, copy=False)
        raw_labels = group[LABEL_COLUMN].to_numpy()
        ramp = pd.to_numeric(group[RAMP_TARGET_COLUMN], errors="coerce").to_numpy(
            dtype=np.float64
        )
        stair = pd.to_numeric(group[STAIR_TARGET_COLUMN], errors="coerce").to_numpy(
            dtype=np.float64
        )
        n_samples = values.shape[0]

        for start in range(0, n_samples - window_size + 1, stride):
            end = start + window_size
            window_labels = raw_labels[start:end]
            if len(set(window_labels.tolist())) != 1:
                n_mixed += 1
                continue
            class_name = normalise_label(window_labels[0])
            if class_name is None:
                n_idle += 1
                continue
            window = values[start:end]
            if not np.isfinite(window).all():
                n_nan += 1
                continue
            # Stored channels-first: models in this repo consume (B, C, T).
            x_windows.append(window.T)
            y_windows.append(CLASS_TO_INDEX[class_name])
            subject_windows.append(subject)
            ramp_windows.append(_nanmean(ramp[start:end]))
            stair_windows.append(_nanmean(stair[start:end]))

    if not x_windows:
        raise RuntimeError(
            f"Segmentation produced no window of {window_size} samples. "
            f"Discarded {n_mixed} mixed-label, {n_idle} idle and {n_nan} "
            "incomplete windows -- try a shorter --window-size."
        )

    LOGGER.info(
        "Segmented %d windows of %d samples (dropped %d mixed-label, %d idle, %d with NaN)",
        len(x_windows),
        window_size,
        n_mixed,
        n_idle,
        n_nan,
    )
    return WindowedData(
        x=np.stack(x_windows).astype(np.float32),
        y=np.asarray(y_windows, dtype=np.int64),
        subjects=np.asarray(subject_windows, dtype=object),
        ramp_incline=np.asarray(ramp_windows, dtype=np.float64),
        stair_height=np.asarray(stair_windows, dtype=np.float64),
        channels=channels,
        window_size=window_size,
    )
