"""Reading harmonised recordings from disk.

The canonical on-disk format is one CSV (or TXT) file per recording with a
header row::

    time,foot_Accel_X,...,trunk_Gyro_Z,gastrocmed,...,label,subject[,ramp_incline,stair_height]

Files produced by the original paper scripts have no header and store the
columns positionally (``index, features..., label, subject``); those are still
readable through ``layout="legacy_imu"``, ``"legacy_emg"`` or
``"legacy_multimodal"``. ``layout="auto"`` sniffs the file and picks the right
reader.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .constants import (
    EMG_CHANNELS,
    IMU_CHANNELS,
    LABEL_COLUMN,
    RAMP_TARGET_COLUMN,
    RECORDING_COLUMN,
    STAIR_TARGET_COLUMN,
    SUBJECT_COLUMN,
    TIME_COLUMN,
)

LOGGER = logging.getLogger(__name__)

DATA_SUFFIXES = (".csv", ".txt")

#: Positional layouts of the headerless files used by the original notebooks.
LEGACY_LAYOUTS = {
    "legacy_imu": list(IMU_CHANNELS),
    "legacy_emg": list(EMG_CHANNELS),
    "legacy_multimodal": list(IMU_CHANNELS) + list(EMG_CHANNELS),
}


def discover_recordings(root: str | Path, pattern: str = "*") -> List[Path]:
    """Recursively list data files under ``root``, sorted for reproducibility."""
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory '{root}' does not exist")
    files = sorted(
        path
        for path in root.rglob(pattern)
        if path.is_file() and path.suffix.lower() in DATA_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"No .csv/.txt recordings found under '{root}' (pattern '{pattern}')"
        )
    return files


def _has_header(path: Path) -> bool:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
    fields = [field.strip().strip('"') for field in first_line.split(",")]
    known = set(IMU_CHANNELS) | set(EMG_CHANNELS) | {
        TIME_COLUMN,
        LABEL_COLUMN,
        SUBJECT_COLUMN,
    }
    return any(field in known for field in fields)


def _infer_legacy_layout(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
    n_columns = len(first_line.split(","))
    # index + features + label (+ subject)
    for name, channels in LEGACY_LAYOUTS.items():
        if n_columns in (len(channels) + 2, len(channels) + 3):
            return name
    raise ValueError(
        f"Cannot infer the legacy layout of '{path.name}': it has {n_columns} "
        "columns, which matches none of the IMU (24), EMG (11) or multimodal "
        "(35) feature sets. Pass --data-layout explicitly."
    )


def _read_legacy(path: Path, layout: str) -> pd.DataFrame:
    channels = LEGACY_LAYOUTS[layout]
    frame = pd.read_csv(path, header=None, engine="c", low_memory=False)
    n_expected_min = len(channels) + 2
    if frame.shape[1] < n_expected_min:
        raise ValueError(
            f"'{path.name}' has {frame.shape[1]} columns but layout '{layout}' "
            f"needs at least {n_expected_min}"
        )
    names = [TIME_COLUMN] + channels + [LABEL_COLUMN]
    if frame.shape[1] > n_expected_min:
        names.append(SUBJECT_COLUMN)
    frame = frame.iloc[:, : len(names)]
    frame.columns = names
    if SUBJECT_COLUMN not in frame.columns:
        # The legacy EMG dumps omit the subject on some rows; fall back to the
        # file name, which encodes the subject in the CAMARGO release.
        frame[SUBJECT_COLUMN] = path.stem.split("_")[0]
    return frame


def read_recording(path: str | Path, layout: str = "auto") -> pd.DataFrame:
    """Read one recording into a dataframe with canonical column names."""
    path = Path(path)
    if layout == "auto":
        layout = "canonical" if _has_header(path) else _infer_legacy_layout(path)

    if layout == "canonical":
        frame = pd.read_csv(path, engine="c", low_memory=False)
        frame.columns = [str(col).strip() for col in frame.columns]
    elif layout in LEGACY_LAYOUTS:
        frame = _read_legacy(path, layout)
    else:
        raise ValueError(
            f"Unknown data layout '{layout}'; expected 'auto', 'canonical' or "
            f"one of {sorted(LEGACY_LAYOUTS)}"
        )

    missing = [col for col in (LABEL_COLUMN, SUBJECT_COLUMN) if col not in frame.columns]
    if missing:
        raise ValueError(f"'{path.name}' is missing required column(s) {missing}")

    frame[RECORDING_COLUMN] = path.stem
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip().str.lower()
    frame[SUBJECT_COLUMN] = frame[SUBJECT_COLUMN].astype(str).str.strip()
    # The legacy files mark rows with no subject as the literal string "none".
    frame = frame[~frame[SUBJECT_COLUMN].str.lower().isin({"none", "nan", ""})]
    for target in (RAMP_TARGET_COLUMN, STAIR_TARGET_COLUMN):
        if target not in frame.columns:
            frame[target] = np.nan
        else:
            frame[target] = pd.to_numeric(frame[target], errors="coerce")
    return frame


def load_recordings(
    root: str | Path,
    channels: Sequence[str],
    layout: str = "auto",
    pattern: str = "*",
    subjects: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Concatenate every recording under ``root``, keeping ``channels``.

    Recordings that do not carry all the requested channels are skipped with a
    warning: this is what lets an IMU-only dump and a multimodal dump live in
    the same folder without breaking an ``--modality imu`` run.
    """
    wanted = set(subjects) if subjects else None
    frames: List[pd.DataFrame] = []
    skipped: List[str] = []

    for path in discover_recordings(root, pattern):
        frame = read_recording(path, layout=layout)
        missing = [channel for channel in channels if channel not in frame.columns]
        if missing:
            skipped.append(f"{path.name} (missing {len(missing)} channel(s))")
            continue
        if wanted is not None:
            frame = frame[frame[SUBJECT_COLUMN].isin(wanted)]
            if frame.empty:
                continue
        keep = [
            *channels,
            LABEL_COLUMN,
            SUBJECT_COLUMN,
            RECORDING_COLUMN,
            RAMP_TARGET_COLUMN,
            STAIR_TARGET_COLUMN,
        ]
        frames.append(frame[keep])

    if skipped:
        LOGGER.warning(
            "Skipped %d recording(s) that do not provide the requested channels: %s",
            len(skipped),
            ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""),
        )
    if not frames:
        raise RuntimeError(
            f"No recording under '{root}' provides the requested channel set "
            f"({len(channels)} channels). Check --modality / --imu-locations."
        )
    return pd.concat(frames, ignore_index=True)
