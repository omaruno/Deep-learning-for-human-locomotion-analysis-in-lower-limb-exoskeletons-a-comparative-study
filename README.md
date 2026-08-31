# Deep Learning for Human Locomotion Analysis in Lower-Limb Exoskeletons

Official implementation of:

> **Deep Learning for Human Locomotion Analysis in Lower-Limb Exoskeletons: A Comparative Study**
>
> O. Coser, C. Tamantini, M. Tortora, L. Furia, R. Sicilia, L. Zollo, P. Soda
>
> *Frontiers in Computer Science*, 2025 — [DOI: 10.3389/fcomp.2025.1597143](https://doi.org/10.3389/fcomp.2025.1597143)

A single configurable pipeline for terrain classification and locomotion
parameter estimation from wearable sensors, comparing **eight time-series
architectures** under **leave-one-subject-out** validation, with **SHAP**-based
sensor-importance analysis.

---

## Table of contents

- [Overview](#overview)
- [Quick start](#quick-start)
- [Usage](#usage)
  - [Choosing a model](#1-choosing-a-model)
  - [Choosing a sensor modality](#2-choosing-a-sensor-modality)
  - [Choosing the temporal window](#3-choosing-the-temporal-window)
  - [Choosing the model size](#4-choosing-the-model-size)
  - [Choosing the task](#5-choosing-the-task)
  - [Running the SHAP analysis](#6-running-the-shap-analysis)
  - [Configuration files](#configuration-files)
- [CLI reference](#cli-reference)
- [Data](#data)
- [Outputs](#outputs)
- [Repository structure](#repository-structure)
- [Reproducing the paper](#reproducing-the-paper)
- [Testing](#testing)
- [Notes on this implementation](#notes-on-this-implementation)
- [Citation](#citation)
- [License](#license)

---

## Overview

The framework addresses two problems for lower-limb exoskeleton control:

**1. Terrain classification** (5 classes)

| Class | Code |
|---|---|
| Level ground | LG |
| Ramp ascent | RA |
| Ramp descent | RD |
| Stair ascent | SA |
| Stair descent | SD |

**2. Locomotion parameter estimation** (regression)

- Ramp slope, in degrees
- Stair height, in millimetres

### Reported results

| Task | Metric | Value |
|---|---|---|
| Terrain classification | Accuracy | 0.94 ± 0.04 |
| Ramp slope regression | MAE | ~1.95° |
| Stair height regression | MAE | ~15.65 mm |
| Inference | Latency | ~1–2 ms |

### Implemented architectures

| `--model` | Architecture | Family |
|---|---|---|
| `cnn` | 1-D convolutional network | Convolutional |
| `lstm` | Long short-term memory | Recurrent |
| `gru` | Gated recurrent unit | Recurrent |
| `cnn_lstm` | Convolutional encoder → LSTM | Hybrid |
| `lstm_cnn` | LSTM encoder → convolutions | Hybrid |
| `xceptiontime` | XceptionTime | Depthwise-separable conv. |
| `tst` | Time Series Transformer | Attention |
| `mamba` | Selective state-space model | SSM |

Best model per task in the paper: **LSTM** for terrain classification and ramp
slope, **CNN-LSTM** for stair height.

### Key finding

An **IMU-only setup using three units (foot, shank, thigh)** matches or
outperforms IMU+EMG configurations — a lighter, cheaper and more practical
wearable system. The SHAP analysis in this repository reproduces the evidence
behind that claim.

---

## Quick start

```bash
git clone https://github.com/omaruno/Deep-learning-for-human-locomotion-analysis-in-lower-limb-exoskeletons-a-comparative-study.git
cd Deep-learning-for-human-locomotion-analysis-in-lower-limb-exoskeletons-a-comparative-study

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch, follow the [official selector](https://pytorch.org/get-started/locally/).
`requirements-paper.txt` is the frozen environment used for the published
experiments, kept for provenance; `requirements.txt` is what you need to run
this code.

The CAMARGO 2021 dataset is not redistributed here. To verify the installation
immediately, generate a synthetic dataset with the same structure:

```bash
python scripts/make_synthetic_dataset.py --output data/synthetic
python main.py --config configs/quick_test.yaml
```

That takes a couple of minutes on CPU and should report a LOSO accuracy well
above chance, confirming that data loading, training, evaluation and reporting
all work.

Then point the pipeline at the real data:

```bash
python preprocessing/prepare_camargo.py --raw-root /path/to/CAMARGO --output data/processed
python main.py --model lstm --modality imu --window-ms 100
```

---

## Usage

Everything runs through **one entry point**, `main.py`. The five choices the
project is built around are five flags.

```bash
python main.py --model lstm --modality imu --window-ms 100 --model-size base --shap
```

### 1. Choosing a model

```bash
python main.py --model cnn
python main.py --model lstm
python main.py --model gru
python main.py --model cnn_lstm
python main.py --model lstm_cnn
python main.py --model xceptiontime
python main.py --model tst
python main.py --model mamba

python main.py --list-models       # print all models and size presets
```

### 2. Choosing a sensor modality

The dataloader assembles the input channels from the modality you pick:

```bash
python main.py --modality imu          # 24 channels: 4 IMUs x (3 accel + 3 gyro)
python main.py --modality emg          # 11 channels: surface EMG
python main.py --modality multimodal   # 35 channels: IMU + EMG
```

You can also restrict the sensor set — this is how the paper's minimal
three-IMU configuration is evaluated:

```bash
python main.py --modality imu --imu-locations foot shank thigh     # 18 channels
python main.py --modality emg --emg-muscles soleus gastrocmed      # 2 channels
```

### 3. Choosing the temporal window

In milliseconds (converted using `--sampling-rate`, default 1000 Hz):

```bash
python main.py --window-ms 100      # the paper's setting -> 100 samples
python main.py --window-ms 250
```

Or directly in samples, with optional overlap:

```bash
python main.py --window-size 200
python main.py --window-size 200 --stride 50    # 75% overlap
```

Windows are non-overlapping by default and never span a terrain transition.

### 4. Choosing the model size

One knob scales every architecture consistently:

| `--model-size` | width | depth | hidden | d_model | layers | heads |
|---|---|---|---|---|---|---|
| `tiny` | 32 | 2 | 32 | 32 | 1 | 2 |
| `small` | 64 | 3 | 64 | 64 | 2 | 4 |
| `base` *(paper)* | 128 | 3 | 128 | 128 | 3 | 8 |
| `large` | 256 | 4 | 256 | 256 | 4 | 8 |

```bash
python main.py --model tst --model-size large
```

Individual fields can be overridden:

```bash
python main.py --model lstm --model-size base --hidden 256 --n-layers 3
python main.py --model tst --d-model 192 --n-heads 6 --dropout 0.3
```

### 5. Choosing the task

```bash
python main.py --task classification      # 5-class terrain (default)
python main.py --task ramp_slope          # regression, degrees
python main.py --task stair_height        # regression, millimetres
```

Regression runs automatically keep only the relevant windows (ramp windows for
`ramp_slope`, stair windows for `stair_height`), switch the loss to MSE and
report MAE / RMSE / R².

### 6. Running the SHAP analysis

Add `--shap` to explain the trained model after training:

```bash
python main.py --model lstm --modality imu --shap
python main.py --model lstm --modality multimodal --shap \
    --shap-background 200 --shap-samples 500 --shap-folds 3
```

This runs `shap.GradientExplainer`, aggregates the absolute attributions over
samples and over the time axis, and writes per fold:

- `shap_channel_importance.csv` — every channel ranked
- `shap_channel_importance.png` — bar chart coloured by sensor, with a dashed mean line per sensor
- `shap_sensor_importance.png` — importance averaged per physical device
- `shap_values.npy` — the raw attributions

plus a `shap_summary.csv` at the run root. SHAP is expensive, so it defaults to
the first fold only (`--shap-folds`).

### Configuration files

Any run can be described in YAML, and CLI flags override the file:

```bash
python main.py --config configs/paper_classification.yaml
python main.py --config configs/paper_classification.yaml --model tst --model-size large
```

| Config | Purpose |
|---|---|
| `configs/paper_classification.yaml` | Terrain classification, IMU, LOSO |
| `configs/paper_ramp_slope.yaml` | Ramp-slope regression |
| `configs/paper_stair_height.yaml` | Stair-height regression |
| `configs/shap_sensor_selection.yaml` | Multimodal run with SHAP enabled |
| `configs/quick_test.yaml` | Fast check on the synthetic dataset |

---

## CLI reference

`python main.py --help` prints the full list. The most useful flags:

**Data**

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/processed` | Folder with the processed recordings |
| `--modality` | `imu` | `imu` \| `emg` \| `multimodal` |
| `--task` | `classification` | `classification` \| `ramp_slope` \| `stair_height` |
| `--window-size` | `100` | Window length, in samples |
| `--window-ms` | — | Window length in ms (overrides `--window-size`) |
| `--stride` | window size | Hop between windows |
| `--sampling-rate` | `1000` | Recording sampling rate, in Hz |
| `--imu-locations` | all four | Subset of `foot shank thigh trunk` |
| `--emg-muscles` | all eleven | Subset of the EMG channels |
| `--data-layout` | `auto` | `canonical` or `legacy_imu`/`legacy_emg`/`legacy_multimodal` |

**Model**

| Flag | Default | Description |
|---|---|---|
| `--model` | `lstm` | Architecture |
| `--model-size` | `base` | `tiny` \| `small` \| `base` \| `large` |
| `--width`, `--depth`, `--hidden`, `--d-model`, `--n-layers`, `--n-heads`, `--head-dim` | preset | Per-field overrides |
| `--dropout` | `0.2` | Dropout probability |

**Training**

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `100` | Maximum epochs per fold |
| `--batch-size` | `64` | |
| `--learning-rate` / `--lr` | `0.001` | Adam learning rate |
| `--patience` | `10` | Early-stopping patience (`0` disables) |
| `--val-fraction` | `0.2` | Share of training **subjects** used for validation |
| `--class-weights` | off | Inverse-frequency class weighting |

**Validation**

| Flag | Default | Description |
|---|---|---|
| `--evaluation` | `loso` | `loso` (paper protocol) or `holdout` |
| `--subjects` | all | Restrict the LOSO folds, e.g. `--subjects AB06 AB07` |
| `--max-folds` | all | Stop after N folds (quick runs) |

**Explainability**

| Flag | Default | Description |
|---|---|---|
| `--shap` | off | Run the SHAP analysis after training |
| `--shap-background` | `100` | Background windows |
| `--shap-samples` | `200` | Windows to explain |
| `--shap-folds` | `1` | How many folds to explain |

**Runtime**

| Flag | Default | Description |
|---|---|---|
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:0` \| `mps` |
| `--seed` | `42` | |
| `--output-dir` | `outputs` | Where runs are written |
| `--run-name` | auto | Defaults to `<model>_<modality>_w<window>_<size>` |
| `--cache-dir` | `outputs/cache` | Window cache (`--cache-dir ''` disables) |
| `--no-checkpoints`, `--no-latency` | — | Skip saving weights / timing inference |

---

## Data

### Dataset

Experiments use the **CAMARGO 2021** dataset:

- 21 able-bodied subjects
- 4 IMUs (trunk, thigh, shank, foot)
- 11 surface EMG sensors
- Multiple ramp inclinations and stair heights

> Camargo et al., *A comprehensive, open-source dataset of lower limb
> biomechanics in multiple conditions of stairs, ramps, and level-ground
> ambulation and transitions*, Journal of Biomechanics, 2021.

The dataset is not redistributed here — download it from the original source.

### Preprocessing

```bash
python preprocessing/prepare_camargo.py \
    --raw-root /path/to/CAMARGO \
    --output data/processed \
    --sampling-rate 1000
```

This resamples the IMU and EMG streams onto one common grid, attaches the
activity labels from the `conditions` files and the ramp/stair targets, and
writes one CSV per trial. See [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for
the exact column contract, the label mapping and the windowing rules — and read
the assumptions documented at the top of the script, since the layout of the
ramp inclination and stair height varies between dataset releases.

### Your own data

Any dataset works as long as it follows the format in
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md): one CSV per recording with a
header, the channel columns, a `label` column and a `subject` column. Headerless
files from the original paper scripts are also readable via `--data-layout`.

---

## Outputs

Each run writes a self-contained folder:

```
outputs/lstm_imu_w100_base/
├── config.yaml              # exact configuration of this run
├── results.json             # per-fold metrics + mean/std summary + SHAP report
├── folds.csv                # flat per-fold table
├── shap_summary.csv         # sensor importance per fold (with --shap)
├── run.log
├── fold_00_AB06/
│   ├── fold.json            # metrics, loss history, scaler statistics
│   ├── model.pt             # weights + config + channel names + scaler
│   └── shap/                # figures and rankings (with --shap)
└── fold_01_AB07/ ...
```

Metrics reported per fold and aggregated as mean ± std:

- **Classification** — accuracy, weighted precision/recall/F1, macro-F1, per-class F1, confusion matrix
- **Regression** — MAE, MSE, RMSE, R²
- **Both** — inference latency (mean, std, p95) at batch size 1

---

## Repository structure

```
.
├── main.py                       # single entry point
├── configs/                      # ready-made experiment configurations
├── locomotion/
│   ├── cli.py                    # argument parsing
│   ├── config.py                 # ExperimentConfig dataclass
│   ├── data/
│   │   ├── constants.py          # channel layout, classes, label mapping
│   │   ├── io.py                 # reading canonical and legacy files
│   │   ├── windowing.py          # window segmentation
│   │   ├── dataset.py            # torch Dataset + per-channel scaler
│   │   └── datamodule.py         # modality/window/task -> LOSO dataloaders
│   ├── models/                   # the eight architectures + registry
│   ├── engine/                   # trainer, early stopping, metrics
│   ├── explain/                  # SHAP analysis
│   ├── experiments/runner.py     # fold loop, reporting
│   └── utils/                    # seeding, logging, serialisation
├── preprocessing/prepare_camargo.py
├── scripts/make_synthetic_dataset.py
├── docs/DATA_FORMAT.md
└── tests/test_pipeline.py
```

---

## Reproducing the paper

```bash
# 1. Preprocess
python preprocessing/prepare_camargo.py --raw-root /path/to/CAMARGO --output data/processed

# 2. Terrain classification, all eight models, IMU, LOSO over 21 subjects
for model in cnn lstm gru cnn_lstm lstm_cnn xceptiontime tst mamba; do
    python main.py --config configs/paper_classification.yaml --model "$model"
done

# 3. Regression
python main.py --config configs/paper_ramp_slope.yaml
python main.py --config configs/paper_stair_height.yaml

# 4. Modality comparison
python main.py --model lstm --modality imu
python main.py --model lstm --modality emg
python main.py --model lstm --modality multimodal

# 5. SHAP sensor-importance analysis
python main.py --config configs/shap_sensor_selection.yaml

# 6. Minimal sensor configuration suggested by SHAP (no trunk IMU)
python main.py --model lstm --modality imu --imu-locations foot shank thigh
```

Full LOSO over 21 subjects is 21 trainings per configuration; a GPU is strongly
recommended (`--device cuda`). Use `--max-folds` while iterating.

---

## Testing

```bash
pip install pytest
pytest -q
```

The suite (47 tests) covers channel selection and label normalisation, window
segmentation, LOSO split disjointness, a forward pass of every model at every
output head and several window lengths, the metrics, CLI parsing, and two full
end-to-end runs (one with SHAP) on a generated dataset.

---

## Notes on this implementation

This is a consolidated rewrite of the original experiment notebooks, which
remain available in the git history of this repository (commit `79b569c` and
earlier). Behavioural differences worth knowing:

- **One framework.** The original code mixed Keras and PyTorch across notebooks;
  everything is now PyTorch, so all eight models are trained, evaluated and
  explained through exactly the same code path.
- **No external model dependency.** XceptionTime, TST and Mamba are implemented
  here rather than imported from `tsai`/`mamba_ssm`. Mamba ships a portable
  PyTorch selective scan and runs on CPU; if `mamba-ssm` is installed and CUDA is
  available, its fused kernel is used instead.
- **Standardisation is fitted on training data only**, per channel, per fold.
- **Validation is subject-wise.** Early stopping uses held-out *subjects* from
  the training split, not random windows, so it cannot leak within-subject
  information.
- **Global average pooling** replaces `Flatten` in the convolutional models, and
  pooling is skipped once the sequence gets short — the original 5-stage pooling
  stack collapsed a 100-sample window. Models therefore work at any window length.
- **SHAP aggregates over the whole window** instead of reading a single time
  step, which makes the channel ranking far more stable.

Because of these changes, numbers reproduced with this code may differ slightly
from the published tables.

---

## Citation

```bibtex
@article{coser2025deep,
  title={Deep learning for human locomotion analysis in lower-limb exoskeletons: a comparative study},
  author={Coser, Omar and Tamantini, Christian and Tortora, Matteo and Furia, Leonardo and Sicilia, Rosa and Zollo, Loredana and Soda, Paolo},
  journal={Frontiers in Computer Science},
  volume={7},
  pages={1597143},
  year={2025},
  publisher={Frontiers Media SA}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
