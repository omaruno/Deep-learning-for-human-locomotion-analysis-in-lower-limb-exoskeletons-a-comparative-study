"""Canonical sensor layout, class definitions and label normalisation.

The channel order defined here is the contract between the preprocessing
stage, the data loader and the SHAP analysis: every array produced by this
package has its channels in exactly this order.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# IMU layout: 4 inertial units x (3 accelerometer + 3 gyroscope) axes = 24 ch. #
# --------------------------------------------------------------------------- #
IMU_LOCATIONS: Tuple[str, ...] = ("foot", "shank", "thigh", "trunk")
IMU_AXES: Tuple[str, ...] = (
    "Accel_X",
    "Accel_Y",
    "Accel_Z",
    "Gyro_X",
    "Gyro_Y",
    "Gyro_Z",
)
IMU_CHANNELS: Tuple[str, ...] = tuple(
    f"{location}_{axis}" for location in IMU_LOCATIONS for axis in IMU_AXES
)

# Short labels used on SHAP figures, where 24 long names do not fit.
IMU_SHORT_NAMES: Dict[str, str] = {"foot": "f", "shank": "s", "thigh": "th", "trunk": "tr"}
AXIS_SHORT_NAMES: Dict[str, str] = {
    "Accel_X": "A_X",
    "Accel_Y": "A_Y",
    "Accel_Z": "A_Z",
    "Gyro_X": "G_X",
    "Gyro_Y": "G_Y",
    "Gyro_Z": "G_Z",
}

# --------------------------------------------------------------------------- #
# EMG layout: the 11 surface electrodes of the CAMARGO 2021 protocol.          #
# --------------------------------------------------------------------------- #
EMG_CHANNELS: Tuple[str, ...] = (
    "gastrocmed",
    "tibialisanterior",
    "soleus",
    "vastusmedialis",
    "vastuslateralis",
    "rectusfemoris",
    "bicepsfemoris",
    "semitendinosus",
    "gracilis",
    "gluteusmedius",
    "rightexternaloblique",
)

# --------------------------------------------------------------------------- #
# Metadata / target columns.                                                   #
# --------------------------------------------------------------------------- #
TIME_COLUMN = "time"
LABEL_COLUMN = "label"
SUBJECT_COLUMN = "subject"
RECORDING_COLUMN = "recording"
RAMP_TARGET_COLUMN = "ramp_incline"   # degrees
STAIR_TARGET_COLUMN = "stair_height"  # millimetres

METADATA_COLUMNS: Tuple[str, ...] = (
    TIME_COLUMN,
    LABEL_COLUMN,
    SUBJECT_COLUMN,
    RECORDING_COLUMN,
    RAMP_TARGET_COLUMN,
    STAIR_TARGET_COLUMN,
)

# --------------------------------------------------------------------------- #
# Terrain classes. The integer order matches the original paper code.          #
# --------------------------------------------------------------------------- #
CLASS_NAMES: Tuple[str, ...] = (
    "stairascent",
    "stairdescent",
    "rampascent",
    "rampdescent",
    "levelground",
)
CLASS_DISPLAY_NAMES: Tuple[str, ...] = ("SA", "SD", "RA", "RD", "LG")
CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Rows carrying these labels are dropped before windowing.
IGNORED_LABELS = frozenset({"idle", "", "nan", "none"})

# Sampling rate (Hz) of the harmonised recordings. CAMARGO streams are
# resampled to a common grid during preprocessing, so one sample == one
# millisecond and a 100 ms window == 100 samples (the paper's setting).
DEFAULT_SAMPLING_RATE_HZ = 1000

# Modalities exposed on the command line.
MODALITIES: Tuple[str, ...] = ("imu", "emg", "multimodal")


def imu_channels(locations: Sequence[str] | None = None) -> List[str]:
    """IMU channel names, optionally restricted to a subset of body locations."""
    selected = tuple(locations) if locations else IMU_LOCATIONS
    unknown = [loc for loc in selected if loc not in IMU_LOCATIONS]
    if unknown:
        raise ValueError(
            f"Unknown IMU location(s) {unknown}; available: {list(IMU_LOCATIONS)}"
        )
    # Keep the canonical order regardless of the order given by the user.
    return [
        f"{location}_{axis}"
        for location in IMU_LOCATIONS
        if location in selected
        for axis in IMU_AXES
    ]


def emg_channels(muscles: Sequence[str] | None = None) -> List[str]:
    """EMG channel names, optionally restricted to a subset of muscles."""
    selected = tuple(muscles) if muscles else EMG_CHANNELS
    unknown = [m for m in selected if m not in EMG_CHANNELS]
    if unknown:
        raise ValueError(
            f"Unknown EMG channel(s) {unknown}; available: {list(EMG_CHANNELS)}"
        )
    return [m for m in EMG_CHANNELS if m in selected]


def channels_for_modality(
    modality: str,
    imu_locations: Sequence[str] | None = None,
    emg_muscles: Sequence[str] | None = None,
) -> List[str]:
    """Resolve ``--modality`` (+ optional sensor subsets) into channel names."""
    modality = modality.lower()
    if modality == "imu":
        return imu_channels(imu_locations)
    if modality == "emg":
        return emg_channels(emg_muscles)
    if modality == "multimodal":
        return imu_channels(imu_locations) + emg_channels(emg_muscles)
    raise ValueError(f"Unknown modality '{modality}'; available: {list(MODALITIES)}")


def short_channel_names(channels: Sequence[str]) -> List[str]:
    """Compact channel names for plots (``foot_Gyro_Y`` -> ``f_G_Y``)."""
    short: List[str] = []
    for channel in channels:
        location, _, axis = channel.partition("_")
        if location in IMU_SHORT_NAMES and axis in AXIS_SHORT_NAMES:
            short.append(f"{IMU_SHORT_NAMES[location]}_{AXIS_SHORT_NAMES[axis]}")
        else:
            short.append(channel[:12])
    return short


def channel_group(channel: str) -> str:
    """Sensor group a channel belongs to (``foot``/``shank``/.../``emg``)."""
    location = channel.split("_", 1)[0]
    return location if location in IMU_LOCATIONS else "emg"


def normalise_label(raw_label: str) -> str | None:
    """Map a raw CAMARGO activity string onto one of :data:`CLASS_NAMES`.

    Transition labels such as ``walk-stairascent`` or ``stairascent-walk``
    are assigned to the terrain they describe, mirroring the mapping used in
    the paper. ``idle`` (and empty) labels return ``None`` so that the caller
    can drop those samples.
    """
    label = str(raw_label).strip().lower()
    if label in IGNORED_LABELS:
        return None
    for token in label.split("-"):
        if token in CLASS_TO_INDEX:
            return token
    # Walking, standing and turning on flat ground all collapse to level ground.
    return "levelground"
