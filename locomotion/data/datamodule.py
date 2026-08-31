"""End-to-end data pipeline: raw files -> windows -> LOSO dataloaders.

This is the single place where ``--modality``, ``--window-size`` and ``--task``
are turned into tensors, so every model in the repository sees exactly the same
inputs.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import CLASS_TO_INDEX, NUM_CLASSES, channels_for_modality
from .dataset import ChannelScaler, WindowDataset
from .io import discover_recordings, load_recordings
from .windowing import WindowedData, segment_dataframe

LOGGER = logging.getLogger(__name__)

#: Regression tasks, with the windows they are computed on and their target.
REGRESSION_TASKS = {
    "ramp_slope": {
        "classes": ("rampascent", "rampdescent"),
        "attribute": "ramp_incline",
        "unit": "deg",
    },
    "stair_height": {
        "classes": ("stairascent", "stairdescent"),
        "attribute": "stair_height",
        "unit": "mm",
    },
}
TASKS: Tuple[str, ...] = ("classification", *REGRESSION_TASKS)


class LocomotionDataModule:
    """Loads, windows, caches and splits the locomotion recordings."""

    def __init__(
        self,
        data_dir,
        modality: str = "imu",
        window_size: int = 100,
        stride: int | None = None,
        task: str = "classification",
        imu_locations: Sequence[str] | None = None,
        emg_muscles: Sequence[str] | None = None,
        data_layout: str = "auto",
        file_pattern: str = "*",
        batch_size: int = 64,
        num_workers: int = 0,
        val_fraction: float = 0.2,
        cache_dir=None,
        seed: int = 42,
    ):
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'; available: {list(TASKS)}")
        self.data_dir = Path(data_dir)
        self.modality = modality
        self.window_size = int(window_size)
        self.stride = int(stride) if stride else int(window_size)
        self.task = task
        self.channels = channels_for_modality(modality, imu_locations, emg_muscles)
        self.data_layout = data_layout
        self.file_pattern = file_pattern
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_fraction = float(val_fraction)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.seed = int(seed)

        self._windows: WindowedData | None = None
        self._targets: np.ndarray | None = None

    # ------------------------------------------------------------- properties #
    @property
    def is_regression(self) -> bool:
        return self.task in REGRESSION_TASKS

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_outputs(self) -> int:
        return 1 if self.is_regression else NUM_CLASSES

    @property
    def target_unit(self) -> str:
        return REGRESSION_TASKS[self.task]["unit"] if self.is_regression else ""

    @property
    def windows(self) -> WindowedData:
        if self._windows is None:
            raise RuntimeError("Call setup() before accessing the data")
        return self._windows

    @property
    def targets(self) -> np.ndarray:
        if self._targets is None:
            raise RuntimeError("Call setup() before accessing the data")
        return self._targets

    # ------------------------------------------------------------------ setup #
    def _source_fingerprint(self) -> str:
        """Name/size/mtime of every input file, so edited data invalidates the cache."""
        try:
            files = discover_recordings(self.data_dir, self.file_pattern)
        except (FileNotFoundError, OSError):
            return "unavailable"
        return ";".join(
            f"{path.name}:{path.stat().st_size}:{int(path.stat().st_mtime)}"
            for path in files
        )

    def _cache_path(self):
        if self.cache_dir is None:
            return None
        key = "|".join(
            [
                str(self.data_dir.resolve()),
                self.file_pattern,
                self.data_layout,
                ",".join(self.channels),
                str(self.window_size),
                str(self.stride),
                self._source_fingerprint(),
            ]
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        name = f"windows_{self.modality}_w{self.window_size}_{digest}.npz"
        return self.cache_dir / name

    def setup(self) -> "LocomotionDataModule":
        """Load and window the data (cached on disk between runs)."""
        if self._windows is not None:
            return self

        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists():
            LOGGER.info("Loading cached windows from %s", cache_path)
            blob = np.load(cache_path, allow_pickle=True)
            windows = WindowedData(
                x=blob["x"],
                y=blob["y"],
                subjects=blob["subjects"],
                ramp_incline=blob["ramp_incline"],
                stair_height=blob["stair_height"],
                channels=list(blob["channels"]),
                window_size=int(blob["window_size"]),
            )
        else:
            LOGGER.info(
                "Loading %s recordings from %s (%d channels)",
                self.modality,
                self.data_dir,
                len(self.channels),
            )
            frame = load_recordings(
                self.data_dir,
                channels=self.channels,
                layout=self.data_layout,
                pattern=self.file_pattern,
            )
            windows = segment_dataframe(
                frame,
                channels=self.channels,
                window_size=self.window_size,
                stride=self.stride,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    x=windows.x,
                    y=windows.y,
                    subjects=windows.subjects,
                    ramp_incline=windows.ramp_incline,
                    stair_height=windows.stair_height,
                    channels=np.asarray(windows.channels, dtype=object),
                    window_size=windows.window_size,
                )
                LOGGER.info("Cached windows to %s", cache_path)

        self._windows, self._targets = self._apply_task(windows)
        LOGGER.info(
            "Task '%s': %d windows, %d subjects, %d channels x %d samples",
            self.task,
            len(self._windows),
            len(self._windows.subject_ids()),
            self._windows.n_channels,
            self._windows.window_size,
        )
        return self

    def _apply_task(self, windows: WindowedData):
        """Filter windows for the selected task and build its target vector."""
        if not self.is_regression:
            return windows, windows.y

        spec = REGRESSION_TASKS[self.task]
        keep_classes = [CLASS_TO_INDEX[name] for name in spec["classes"]]
        target = getattr(windows, spec["attribute"])
        mask = np.isin(windows.y, keep_classes) & np.isfinite(target)
        if not mask.any():
            raise RuntimeError(
                f"Task '{self.task}' needs windows of classes {spec['classes']} with a "
                f"finite '{spec['attribute']}' column, but none were found. Make sure "
                "the preprocessing step wrote that column."
            )
        index = np.flatnonzero(mask)
        return windows.select(index), target[index].astype(np.float32)

    # ----------------------------------------------------------------- splits #
    def subject_ids(self) -> List[str]:
        return self.windows.subject_ids()

    def loso_splits(
        self, subjects: Sequence[str] | None = None
    ) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        """Yield ``(held_out_subject, train_index, test_index)`` per LOSO fold."""
        all_subjects = self.subject_ids()
        selected = list(subjects) if subjects else all_subjects
        unknown = [s for s in selected if s not in all_subjects]
        if unknown:
            raise ValueError(f"Unknown subject(s) {unknown}; available: {all_subjects}")
        subject_array = self.windows.subjects
        for subject in selected:
            test_index = np.flatnonzero(subject_array == subject)
            train_index = np.flatnonzero(subject_array != subject)
            if len(test_index) == 0 or len(train_index) == 0:
                LOGGER.warning("Skipping subject %s: empty train or test split", subject)
                continue
            yield subject, train_index, test_index

    def holdout_split(self, test_fraction: float = 0.2):
        """A single subject-wise hold-out split, for quick experiments."""
        subjects = self.subject_ids()
        rng = np.random.default_rng(self.seed)
        n_test = max(1, int(round(len(subjects) * test_fraction)))
        test_subjects = list(rng.permutation(np.asarray(subjects, dtype=object))[:n_test])
        mask = np.isin(self.windows.subjects, test_subjects)
        name = "+".join(sorted(str(s) for s in test_subjects))
        return name, np.flatnonzero(~mask), np.flatnonzero(mask)

    def _split_validation(self, train_index: np.ndarray):
        """Carve a subject-wise validation set out of the training index."""
        if self.val_fraction <= 0:
            return train_index, np.empty(0, dtype=np.int64)

        rng = np.random.default_rng(self.seed)
        subject_array = self.windows.subjects[train_index]
        train_subjects = sorted(set(subject_array.tolist()))
        if len(train_subjects) < 2:
            # Not enough subjects to hold one out: fall back to a random split.
            shuffled = rng.permutation(train_index)
            n_val = max(1, int(round(len(shuffled) * self.val_fraction)))
            return shuffled[n_val:], shuffled[:n_val]

        n_val = max(1, int(round(len(train_subjects) * self.val_fraction)))
        subjects_pool = np.asarray(train_subjects, dtype=object)
        val_subjects = list(rng.permutation(subjects_pool)[:n_val])
        mask = np.isin(subject_array, val_subjects)
        return train_index[~mask], train_index[mask]

    # ------------------------------------------------------------ dataloaders #
    def dataloaders(self, train_index: np.ndarray, test_index: np.ndarray):
        """Build train / validation / test loaders and the fitted scaler."""
        fit_index, val_index = self._split_validation(train_index)
        x = self.windows.x
        y = self.targets

        scaler = ChannelScaler.fit(x[fit_index])
        generator = torch.Generator().manual_seed(self.seed)

        def make(index: np.ndarray, shuffle: bool) -> DataLoader:
            dataset = WindowDataset(
                scaler.transform(x[index]), y[index], regression=self.is_regression
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
                generator=generator if shuffle else None,
            )

        train_loader = make(fit_index, shuffle=True)
        val_loader = make(val_index, shuffle=False) if len(val_index) else None
        test_loader = make(test_index, shuffle=False)
        return train_loader, val_loader, test_loader, scaler

    def class_weights(self, train_index: np.ndarray) -> torch.Tensor:
        """Inverse-frequency class weights, for the imbalanced terrain classes."""
        counts = np.bincount(self.windows.y[train_index], minlength=NUM_CLASSES)
        total = counts.sum()
        weights = np.where(
            counts > 0, total / (NUM_CLASSES * np.maximum(counts, 1)), 0.0
        )
        return torch.tensor(weights, dtype=torch.float32)
