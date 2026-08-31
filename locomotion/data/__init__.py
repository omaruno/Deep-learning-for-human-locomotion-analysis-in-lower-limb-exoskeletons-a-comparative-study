"""Data loading, windowing and LOSO splitting."""

from .constants import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    EMG_CHANNELS,
    IMU_CHANNELS,
    IMU_LOCATIONS,
    MODALITIES,
    NUM_CLASSES,
    channels_for_modality,
)
from .datamodule import REGRESSION_TASKS, TASKS, LocomotionDataModule
from .dataset import ChannelScaler, WindowDataset
from .io import load_recordings, read_recording
from .windowing import WindowedData, segment_dataframe

__all__ = [
    "CLASS_DISPLAY_NAMES",
    "CLASS_NAMES",
    "ChannelScaler",
    "EMG_CHANNELS",
    "IMU_CHANNELS",
    "IMU_LOCATIONS",
    "LocomotionDataModule",
    "MODALITIES",
    "NUM_CLASSES",
    "REGRESSION_TASKS",
    "TASKS",
    "WindowDataset",
    "WindowedData",
    "channels_for_modality",
    "load_recordings",
    "read_recording",
    "segment_dataframe",
]
