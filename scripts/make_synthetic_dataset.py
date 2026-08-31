#!/usr/bin/env python
"""Generate a small synthetic dataset in the canonical format.

The CAMARGO 2021 recordings are not redistributed with this repository. This
script writes plausible surrogate data with the same columns, labels and
subject structure so that the whole pipeline -- loading, windowing, LOSO
training, SHAP -- can be run and tested end to end.

    python scripts/make_synthetic_dataset.py --output data/synthetic --subjects 6

The signals are *not* real biomechanics: each terrain class gets its own
frequency, amplitude and offset pattern, with the foot IMU carrying the
strongest class information (mirroring the paper's SHAP finding) so that the
sanity checks produce sensible accuracies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locomotion.data.constants import (  # noqa: E402
    CLASS_NAMES,
    EMG_CHANNELS,
    IMU_CHANNELS,
    IMU_LOCATIONS,
)

# Per-class signal signature: (base frequency in Hz, amplitude, offset).
CLASS_SIGNATURE = {
    "stairascent": (2.4, 1.60, 0.9),
    "stairdescent": (2.2, 1.45, -0.8),
    "rampascent": (1.7, 1.05, 0.45),
    "rampdescent": (1.6, 0.95, -0.40),
    "levelground": (1.9, 0.70, 0.0),
}

#: How strongly each IMU location reflects the terrain class. The foot IMU is
#: the most informative and the trunk the least, as reported in the paper.
LOCATION_GAIN = {"foot": 1.0, "shank": 0.75, "thigh": 0.5, "trunk": 0.18}

RAMP_INCLINES = (5.2, 7.8, 9.2, 11.0, 12.4, 18.0)   # degrees
STAIR_HEIGHTS = (102.0, 127.0, 152.0, 178.0)        # millimetres


def _activity_plan(rng: np.random.Generator, n_bouts: int):
    """A sequence of (class, duration in samples) bouts, with idle in between."""
    plan = []
    for _ in range(n_bouts):
        class_name = CLASS_NAMES[rng.integers(len(CLASS_NAMES))]
        plan.append((class_name, int(rng.integers(400, 900))))
        if rng.random() < 0.35:
            plan.append(("idle", int(rng.integers(80, 200))))
    return plan


def _make_bout(
    class_name: str,
    n_samples: int,
    subject_offset: float,
    sampling_rate: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    time = np.arange(n_samples) / sampling_rate
    frame = pd.DataFrame({"time": time})

    if class_name == "idle":
        frequency, amplitude, offset = 0.2, 0.05, 0.0
    else:
        frequency, amplitude, offset = CLASS_SIGNATURE[class_name]
    phase = rng.uniform(0, 2 * np.pi)

    for channel in IMU_CHANNELS:
        location = channel.split("_")[0]
        gain = LOCATION_GAIN[location]
        harmonic = 1 + (IMU_CHANNELS.index(channel) % 3)
        signal = (
            amplitude * gain * np.sin(2 * np.pi * frequency * harmonic * time + phase)
            + gain * offset
            + subject_offset * 0.1
        )
        noise = rng.normal(0, 0.12 + 0.25 * (1 - gain), n_samples)
        # Gyroscopes are an order of magnitude larger than accelerometers.
        scale = 60.0 if "Gyro" in channel else 1.0
        frame[channel] = (signal + noise) * scale

    for index, muscle in enumerate(EMG_CHANNELS):
        envelope = np.abs(
            amplitude * np.sin(2 * np.pi * frequency * time + phase + index * 0.4)
        )
        frame[muscle] = (envelope * 0.4 + rng.normal(0, 0.08, n_samples)) * 1e-3

    frame["label"] = class_name
    frame["ramp_incline"] = (
        rng.choice(RAMP_INCLINES) * (1 if class_name == "rampascent" else -1)
        if class_name in {"rampascent", "rampdescent"}
        else np.nan
    )
    frame["stair_height"] = (
        rng.choice(STAIR_HEIGHTS)
        if class_name in {"stairascent", "stairdescent"}
        else np.nan
    )
    return frame


def generate(
    output: Path,
    n_subjects: int = 6,
    n_trials: int = 2,
    n_bouts: int = 8,
    sampling_rate: int = 1000,
    seed: int = 42,
) -> list:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = []

    for subject_index in range(n_subjects):
        subject = f"AB{6 + subject_index:02d}"
        subject_offset = rng.normal(0, 1.0)
        for trial in range(n_trials):
            bouts = [
                _make_bout(class_name, duration, subject_offset, sampling_rate, rng)
                for class_name, duration in _activity_plan(rng, n_bouts)
            ]
            frame = pd.concat(bouts, ignore_index=True)
            frame["time"] = np.arange(len(frame)) / sampling_rate
            frame["subject"] = subject
            path = output / f"{subject}_trial{trial + 1}.csv"
            frame.to_csv(path, index=False, float_format="%.6f")
            written.append(path)
            print(f"  wrote {path}  ({len(frame):,} samples)")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--bouts", type=int, default=8, help="activity bouts per trial")
    parser.add_argument("--sampling-rate", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating synthetic recordings in '{args.output}'")
    files = generate(
        args.output,
        n_subjects=args.subjects,
        n_trials=args.trials,
        n_bouts=args.bouts,
        sampling_rate=args.sampling_rate,
        seed=args.seed,
    )
    print(
        f"\nDone: {len(files)} files, {len(IMU_CHANNELS)} IMU + {len(EMG_CHANNELS)} EMG "
        f"channels ({', '.join(IMU_LOCATIONS)})."
    )
    print(f"Try:  python main.py --data-dir {args.output} --model lstm --modality imu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
