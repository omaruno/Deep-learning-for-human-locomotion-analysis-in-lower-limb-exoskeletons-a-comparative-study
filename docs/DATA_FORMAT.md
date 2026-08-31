# Data format

Every script in this repository reads the same canonical format: **one CSV file
per recording (trial), with a header row**. Files may live in nested folders —
the loader searches recursively.

## Columns

| Column | Required | Description |
|---|---|---|
| `time` | no | Time stamp in seconds, restarting at 0 for each recording. Informational only. |
| 24 IMU columns | for `--modality imu` / `multimodal` | See below. |
| 11 EMG columns | for `--modality emg` / `multimodal` | See below. |
| `label` | **yes** | Activity string (see *Labels*). |
| `subject` | **yes** | Subject identifier, e.g. `AB06`. Drives the LOSO splits. |
| `ramp_incline` | for `--task ramp_slope` | Ramp inclination in **degrees** (positive = ascent, negative = descent). `NaN` elsewhere. |
| `stair_height` | for `--task stair_height` | Step height in **millimetres**. `NaN` elsewhere. |

Extra columns are ignored, so you can keep goniometer or force-plate channels
in the same file.

### IMU channels (24)

Four inertial units × 6 axes, named `<location>_<sensor>_<axis>`:

```
foot_Accel_X   foot_Accel_Y   foot_Accel_Z   foot_Gyro_X   foot_Gyro_Y   foot_Gyro_Z
shank_Accel_X  shank_Accel_Y  shank_Accel_Z  shank_Gyro_X  shank_Gyro_Y  shank_Gyro_Z
thigh_Accel_X  thigh_Accel_Y  thigh_Accel_Z  thigh_Gyro_X  thigh_Gyro_Y  thigh_Gyro_Z
trunk_Accel_X  trunk_Accel_Y  trunk_Accel_Z  trunk_Gyro_X  trunk_Gyro_Y  trunk_Gyro_Z
```

This is the naming used by the CAMARGO 2021 release, and the order is the
contract used by the SHAP plots. `--imu-locations` selects a subset of the four
locations while preserving this order.

### EMG channels (11)

```
gastrocmed  tibialisanterior  soleus  vastusmedialis  vastuslateralis  rectusfemoris
bicepsfemoris  semitendinosus  gracilis  gluteusmedius  rightexternaloblique
```

## Labels

`label` holds the raw activity string. It is normalised to five terrain
classes; transition labels are assigned to the terrain they describe:

| Raw label | Class | Index |
|---|---|---|
| `stairascent`, `walk-stairascent`, `stairascent-walk` | `stairascent` (SA) | 0 |
| `stairdescent`, `walk-stairdescent`, `stairdescent-walk` | `stairdescent` (SD) | 1 |
| `rampascent`, `walk-rampascent`, `rampascent-walk` | `rampascent` (RA) | 2 |
| `rampdescent`, `walk-rampdescent`, `rampdescent-walk` | `rampdescent` (RD) | 3 |
| `walk`, `stand`, `turn`, anything else | `levelground` (LG) | 4 |
| `idle` | *dropped* | — |

## Sampling rate and windows

All streams must share one sampling grid. The preprocessing script resamples
IMU and EMG onto a common rate (`--sampling-rate`, default 1000 Hz), which makes
the paper's 100 ms window exactly 100 samples. If your data is at another rate,
pass `--sampling-rate` to `main.py` so that `--window-ms` converts correctly, or
just use `--window-size` in samples.

## Windowing rules

A window is kept only if:

1. it lies entirely inside one recording,
2. every sample carries the same raw label,
3. the label is not `idle`,
4. it contains no missing values.

Windows are non-overlapping by default (`--stride` defaults to `--window-size`).

## Legacy (headerless) files

Files produced by the original paper scripts have no header and store columns
positionally as `index, features..., label, subject`. They are still readable:

```bash
python main.py --data-dir data/legacy --data-layout legacy_imu    # 24 features
python main.py --data-dir data/legacy --data-layout legacy_emg    # 11 features
python main.py --data-dir data/legacy --data-layout legacy_multimodal  # 35
```

`--data-layout auto` (the default) sniffs the header and, when absent, infers
the layout from the column count. Legacy files carry no regression targets, so
`--task ramp_slope` / `--task stair_height` need the canonical format.

## Example

```csv
time,foot_Accel_X,...,trunk_Gyro_Z,gastrocmed,...,rightexternaloblique,label,subject,ramp_incline,stair_height
0.000,0.981,...,-1.204,0.000132,...,0.000087,rampascent,AB06,7.8,
0.001,0.977,...,-1.198,0.000129,...,0.000091,rampascent,AB06,7.8,
```

Run `python scripts/make_synthetic_dataset.py` to generate a valid example
dataset you can inspect.
