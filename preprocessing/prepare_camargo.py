#!/usr/bin/env python
"""Convert the CAMARGO 2021 dataset into the canonical format of this repository.

Expected input layout (the CSV distribution of the dataset)::

    <raw-root>/
        AB06/
            10_09_18/
                levelground/
                    conditions/  levelground_ccw_normal_01_01.csv
                    emg/         levelground_ccw_normal_01_01.csv
                    imu/         levelground_ccw_normal_01_01.csv
                ramp/ ...
                stair/ ...
                treadmill/ ...
        AB07/ ...

Each ``imu``/``emg``/``conditions`` file shares a ``Header`` column holding the
time stamp in seconds. IMU and EMG are sampled at different rates, so both are
resampled onto a common grid (``--sampling-rate``, 1000 Hz by default, which
makes the paper's 100 ms window exactly 100 samples).

Output: one CSV per trial under ``--output``, with the columns documented in
``docs/DATA_FORMAT.md``.

    python preprocessing/prepare_camargo.py --raw-root /path/to/CAMARGO --output data/processed

Notes and assumptions:
  * The dataset ships as MATLAB ``.mat`` tables; this script reads the CSV
    export. Convert the ``.mat`` files first if needed (see the README).
  * The ramp inclination and the stair height are taken from the ``conditions``
    file when it provides them, otherwise they are parsed from the trial
    file name (e.g. ``ramp_1_r_...`` / ``stair_1_l_...``), using
    ``--ramp-map`` / ``--stair-map``. Check those mappings against the release
    notes of the dataset version you downloaded.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locomotion.data.constants import (  # noqa: E402
    EMG_CHANNELS,
    IMU_CHANNELS,
    LABEL_COLUMN,
    RAMP_TARGET_COLUMN,
    STAIR_TARGET_COLUMN,
    SUBJECT_COLUMN,
    TIME_COLUMN,
)

LOGGER = logging.getLogger("prepare_camargo")

TIME_ALIASES = ("Header", "header", "time", "Time")
LABEL_ALIASES = ("Label", "label")
ACTIVITY_FOLDERS = ("levelground", "ramp", "stair", "treadmill")

#: Ramp trial index -> inclination in degrees (CAMARGO ramp protocol).
DEFAULT_RAMP_MAP: Dict[str, float] = {
    "1": 5.2, "2": 7.8, "3": 9.2, "4": 11.0, "5": 12.4, "6": 18.0,
}
#: Stair trial index -> step height in millimetres (4 in, 5 in, 6 in, 7 in).
DEFAULT_STAIR_MAP: Dict[str, float] = {
    "1": 102.0, "2": 127.0, "3": 152.0, "4": 178.0,
}


def _find_time_column(frame: pd.DataFrame, path: Path) -> str:
    for alias in TIME_ALIASES:
        if alias in frame.columns:
            return alias
    raise ValueError(
        f"'{path}' has no time column (looked for {list(TIME_ALIASES)}); "
        f"columns are {list(frame.columns)[:8]}"
    )


def _resample(frame: pd.DataFrame, time_column: str, grid: np.ndarray) -> pd.DataFrame:
    """Linear interpolation of every numeric column onto ``grid``."""
    source_time = frame[time_column].to_numpy(dtype=np.float64)
    order = np.argsort(source_time)
    source_time = source_time[order]
    output = {}
    for column in frame.columns:
        if column == time_column:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        output[column] = np.interp(grid, source_time, values[order])
    return pd.DataFrame(output)


def _resample_labels(
    frame: pd.DataFrame, time_column: str, label_column: str, grid: np.ndarray
) -> np.ndarray:
    """Nearest-previous-sample lookup: labels are categorical, not interpolable."""
    source_time = frame[time_column].to_numpy(dtype=np.float64)
    order = np.argsort(source_time)
    source_time = source_time[order]
    labels = frame[label_column].to_numpy()[order]
    index = np.clip(np.searchsorted(source_time, grid, side="right") - 1, 0, len(labels) - 1)
    return labels[index]


def _trial_targets(
    conditions: pd.DataFrame,
    activity: str,
    trial_name: str,
    ramp_map: Dict[str, float],
    stair_map: Dict[str, float],
) -> Dict[str, float]:
    """Ramp inclination / stair height for one trial."""
    targets = {RAMP_TARGET_COLUMN: np.nan, STAIR_TARGET_COLUMN: np.nan}

    # 1. Prefer an explicit column in the conditions file.
    for column in conditions.columns:
        lowered = column.lower()
        values = pd.to_numeric(conditions[column], errors="coerce").dropna()
        if values.empty:
            continue
        if "ramp" in lowered and ("incline" in lowered or "angle" in lowered or "slope" in lowered):
            targets[RAMP_TARGET_COLUMN] = float(values.median())
        elif "stair" in lowered and "height" in lowered:
            targets[STAIR_TARGET_COLUMN] = float(values.median())

    # 2. Otherwise fall back to the index encoded in the trial file name.
    match = re.search(r"(?:ramp|stair)[_-]?(\d+)", trial_name.lower())
    if match:
        index = match.group(1)
        if activity == "ramp" and np.isnan(targets[RAMP_TARGET_COLUMN]):
            targets[RAMP_TARGET_COLUMN] = ramp_map.get(index, np.nan)
        elif activity == "stair" and np.isnan(targets[STAIR_TARGET_COLUMN]):
            targets[STAIR_TARGET_COLUMN] = stair_map.get(index, np.nan)
    return targets


def _sign_ramp(frame: pd.DataFrame) -> pd.DataFrame:
    """Ramp descent is stored as a negative inclination, ascent as positive."""
    descent = frame[LABEL_COLUMN].astype(str).str.contains("rampdescent")
    frame.loc[descent, RAMP_TARGET_COLUMN] = -frame.loc[descent, RAMP_TARGET_COLUMN].abs()
    ascent = frame[LABEL_COLUMN].astype(str).str.contains("rampascent")
    frame.loc[ascent, RAMP_TARGET_COLUMN] = frame.loc[ascent, RAMP_TARGET_COLUMN].abs()
    return frame


def convert_trial(
    imu_path: Path,
    emg_path: Optional[Path],
    conditions_path: Path,
    subject: str,
    activity: str,
    sampling_rate: int,
    ramp_map: Dict[str, float],
    stair_map: Dict[str, float],
) -> Optional[pd.DataFrame]:
    """Merge the IMU, EMG and condition streams of one trial."""
    imu = pd.read_csv(imu_path)
    conditions = pd.read_csv(conditions_path)

    imu_time_column = _find_time_column(imu, imu_path)
    condition_time_column = _find_time_column(conditions, conditions_path)
    label_column = next((a for a in LABEL_ALIASES if a in conditions.columns), None)
    if label_column is None:
        LOGGER.warning("Skipping %s: the conditions file has no Label column", imu_path.name)
        return None

    missing_imu = [c for c in IMU_CHANNELS if c not in imu.columns]
    if missing_imu:
        LOGGER.warning(
            "Skipping %s: missing %d IMU channel(s), e.g. %s",
            imu_path.name, len(missing_imu), missing_imu[:3],
        )
        return None

    # Common time grid: the overlap of the available streams.
    start = float(imu[imu_time_column].min())
    stop = float(imu[imu_time_column].max())
    emg = None
    if emg_path is not None and emg_path.exists():
        emg = pd.read_csv(emg_path)
        emg_time_column = _find_time_column(emg, emg_path)
        if [c for c in EMG_CHANNELS if c not in emg.columns]:
            LOGGER.warning("%s: EMG channels incomplete, writing IMU only", imu_path.name)
            emg = None
        else:
            start = max(start, float(emg[emg_time_column].min()))
            stop = min(stop, float(emg[emg_time_column].max()))

    if stop - start < 1.0 / sampling_rate:
        LOGGER.warning("Skipping %s: streams do not overlap in time", imu_path.name)
        return None

    grid = np.arange(start, stop, 1.0 / sampling_rate)
    frame = _resample(imu[[imu_time_column, *IMU_CHANNELS]], imu_time_column, grid)
    if emg is not None:
        emg_time_column = _find_time_column(emg, emg_path)
        frame = pd.concat(
            [frame, _resample(emg[[emg_time_column, *EMG_CHANNELS]], emg_time_column, grid)],
            axis=1,
        )

    frame.insert(0, TIME_COLUMN, grid - grid[0])
    frame[LABEL_COLUMN] = _resample_labels(
        conditions, condition_time_column, label_column, grid
    )
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip().str.lower()
    frame[SUBJECT_COLUMN] = subject

    targets = _trial_targets(conditions, activity, imu_path.stem, ramp_map, stair_map)
    frame[RAMP_TARGET_COLUMN] = targets[RAMP_TARGET_COLUMN]
    frame[STAIR_TARGET_COLUMN] = targets[STAIR_TARGET_COLUMN]
    return _sign_ramp(frame)


def convert_dataset(
    raw_root: Path,
    output: Path,
    sampling_rate: int,
    activities: List[str],
    ramp_map: Dict[str, float],
    stair_map: Dict[str, float],
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p for p in raw_root.iterdir() if p.is_dir())
    if not subjects:
        raise FileNotFoundError(f"No subject folder found under '{raw_root}'")

    n_written = 0
    for subject_dir in subjects:
        subject = subject_dir.name
        for imu_dir in sorted(subject_dir.rglob("imu")):
            activity = imu_dir.parent.name
            if activity not in activities:
                continue
            conditions_dir = imu_dir.parent / "conditions"
            emg_dir = imu_dir.parent / "emg"
            if not conditions_dir.is_dir():
                LOGGER.warning("No 'conditions' folder next to %s", imu_dir)
                continue

            for imu_path in sorted(imu_dir.glob("*.csv")):
                conditions_path = conditions_dir / imu_path.name
                if not conditions_path.exists():
                    LOGGER.warning("No conditions file for %s", imu_path.name)
                    continue
                frame = convert_trial(
                    imu_path,
                    emg_dir / imu_path.name,
                    conditions_path,
                    subject=subject,
                    activity=activity,
                    sampling_rate=sampling_rate,
                    ramp_map=ramp_map,
                    stair_map=stair_map,
                )
                if frame is None or frame.empty:
                    continue
                destination = output / f"{subject}_{activity}_{imu_path.stem}.csv"
                frame.to_csv(destination, index=False, float_format="%.6f")
                n_written += 1
                LOGGER.info("wrote %s (%d samples)", destination.name, len(frame))
    return n_written


def _parse_map(text: Optional[str], default: Dict[str, float]) -> Dict[str, float]:
    if not text:
        return dict(default)
    mapping = {}
    for item in text.split(","):
        key, _, value = item.partition(":")
        mapping[key.strip()] = float(value)
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CAMARGO 2021 into the canonical format of this repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-root", type=Path, required=True, help="root of the raw dataset")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--sampling-rate", type=int, default=1000, help="common resampling rate, in Hz"
    )
    parser.add_argument(
        "--activities", nargs="+", default=list(ACTIVITY_FOLDERS), choices=ACTIVITY_FOLDERS
    )
    parser.add_argument(
        "--ramp-map", type=str, default=None, help="e.g. '1:5.2,2:7.8,...' (degrees)"
    )
    parser.add_argument(
        "--stair-map", type=str, default=None, help="e.g. '1:102,2:127,...' (millimetres)"
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)-7s | %(message)s",
    )
    n_written = convert_dataset(
        args.raw_root,
        args.output,
        sampling_rate=args.sampling_rate,
        activities=args.activities,
        ramp_map=_parse_map(args.ramp_map, DEFAULT_RAMP_MAP),
        stair_map=_parse_map(args.stair_map, DEFAULT_STAIR_MAP),
    )
    if n_written == 0:
        LOGGER.error("No trial was converted -- check --raw-root and the folder layout")
        return 1
    print(f"\nConverted {n_written} trials into '{args.output}'")
    print(f"Next:  python main.py --data-dir {args.output} --model lstm --modality imu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
