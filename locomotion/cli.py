"""Command-line interface.

Everything the user can choose -- model, modality, window length, model size,
SHAP -- is declared here and mapped onto :class:`~locomotion.config.ExperimentConfig`.
"""

from __future__ import annotations

import argparse
from typing import List, Sequence

from .config import ExperimentConfig
from .data.constants import EMG_CHANNELS, IMU_LOCATIONS, MODALITIES
from .data.datamodule import TASKS
from .models import MODEL_DESCRIPTIONS, MODEL_SIZES, available_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Deep learning for human locomotion analysis in lower-limb "
            "exoskeletons -- train, evaluate and explain time-series models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --model lstm --modality imu --window-size 100\n"
            "  python main.py --model tst --modality multimodal --model-size large --shap\n"
            "  python main.py --model cnn_lstm --task stair_height --modality imu\n"
        ),
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

    # ------------------------------------------------------------------ meta #
    meta = parser.add_argument_group("informational")
    meta.add_argument(
        "--list-models", action="store_true", help="print the available models and exit"
    )
    meta.add_argument(
        "--config", type=str, default=None, help="YAML file with defaults (CLI wins)"
    )

    # ------------------------------------------------------------------ data #
    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", type=str, help="folder with the processed recordings")
    data.add_argument(
        "--modality",
        type=str,
        choices=MODALITIES,
        help="sensor set fed to the model: IMU only, EMG only, or both",
    )
    data.add_argument(
        "--task", type=str, choices=TASKS, help="terrain classification or a regression target"
    )
    data.add_argument(
        "--window-size", type=int, help="temporal window length, in samples"
    )
    data.add_argument(
        "--window-ms",
        type=float,
        help="temporal window length in milliseconds (overrides --window-size)",
    )
    data.add_argument(
        "--stride", type=int, help="hop between windows in samples (default: no overlap)"
    )
    data.add_argument("--sampling-rate", type=int, help="sampling rate of the recordings, in Hz")
    data.add_argument(
        "--imu-locations",
        nargs="+",
        choices=IMU_LOCATIONS,
        help="restrict the IMU set, e.g. --imu-locations foot shank thigh",
    )
    data.add_argument(
        "--emg-muscles", nargs="+", choices=EMG_CHANNELS, help="restrict the EMG channels"
    )
    data.add_argument(
        "--data-layout",
        type=str,
        choices=("auto", "canonical", "legacy_imu", "legacy_emg", "legacy_multimodal"),
        help="on-disk column layout of the recordings",
    )
    data.add_argument("--file-pattern", type=str, help="glob applied to the data folder")

    # ----------------------------------------------------------------- model #
    model = parser.add_argument_group("model")
    model.add_argument("--model", type=str, choices=available_models(), help="architecture")
    model.add_argument(
        "--model-size",
        type=str,
        choices=list(MODEL_SIZES),
        help="capacity preset ('base' reproduces the paper)",
    )
    model.add_argument("--width", type=int, help="override: convolution filters")
    model.add_argument("--depth", type=int, help="override: convolution blocks")
    model.add_argument("--hidden", type=int, help="override: recurrent hidden size")
    model.add_argument("--d-model", type=int, help="override: TST / Mamba embedding size")
    model.add_argument("--n-layers", type=int, help="override: number of layers")
    model.add_argument("--n-heads", type=int, help="override: attention heads")
    model.add_argument("--head-dim", type=int, help="override: dense head width")
    model.add_argument("--dropout", type=float, help="dropout probability")

    # -------------------------------------------------------------- training #
    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, help="maximum epochs per fold")
    training.add_argument("--batch-size", type=int)
    training.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
    training.add_argument("--weight-decay", type=float)
    training.add_argument("--patience", type=int, help="early-stopping patience (0 disables)")
    training.add_argument("--grad-clip", type=float, help="gradient-norm clipping (0 disables)")
    training.add_argument(
        "--val-fraction", type=float, help="share of training subjects used for validation"
    )
    training.add_argument(
        "--class-weights",
        action="store_true",
        default=None,
        help="weight the loss by inverse class frequency",
    )
    training.add_argument("--num-workers", type=int, help="dataloader worker processes")

    # ------------------------------------------------------------ validation #
    validation = parser.add_argument_group("validation")
    validation.add_argument(
        "--evaluation",
        type=str,
        choices=("loso", "holdout"),
        help="leave-one-subject-out (paper protocol) or a single hold-out split",
    )
    validation.add_argument("--subjects", nargs="+", help="restrict LOSO to these subjects")
    validation.add_argument("--max-folds", type=int, help="stop after N folds (quick runs)")
    validation.add_argument("--test-fraction", type=float, help="hold-out test share")

    # ------------------------------------------------------------------ shap #
    shap_group = parser.add_argument_group("explainability")
    shap_group.add_argument(
        "--shap",
        action="store_true",
        default=None,
        help="run the SHAP sensor-importance analysis after training",
    )
    shap_group.add_argument("--shap-background", type=int, help="background windows for SHAP")
    shap_group.add_argument("--shap-samples", type=int, help="windows to explain")
    shap_group.add_argument("--shap-folds", type=int, help="how many folds to explain")

    # --------------------------------------------------------------- runtime #
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", type=str, help="auto | cpu | cuda | cuda:0 | mps")
    runtime.add_argument("--seed", type=int)
    runtime.add_argument("--deterministic", action="store_true", default=None)
    runtime.add_argument("--output-dir", type=str, help="where runs are written")
    runtime.add_argument("--run-name", type=str, help="subfolder name (auto-generated by default)")
    runtime.add_argument("--cache-dir", type=str, help="window cache location ('' disables)")
    runtime.add_argument(
        "--no-checkpoints",
        dest="save_checkpoints",
        action="store_false",
        default=None,
        help="do not save model weights",
    )
    runtime.add_argument(
        "--no-latency",
        dest="measure_latency",
        action="store_false",
        default=None,
        help="skip the inference-latency measurement",
    )
    runtime.add_argument(
        "--log-level", type=str, choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def format_model_table() -> str:
    lines = ["Available models (--model):", ""]
    width = max(len(name) for name in MODEL_DESCRIPTIONS)
    for name, description in MODEL_DESCRIPTIONS.items():
        lines.append(f"  {name:<{width}}  {description}")
    lines += ["", "Available sizes (--model-size):", ""]
    for name, spec in MODEL_SIZES.items():
        lines.append(
            f"  {name:<6}  width={spec.width:<4} depth={spec.depth}  hidden={spec.hidden:<4} "
            f"d_model={spec.d_model:<4} n_layers={spec.n_layers} n_heads={spec.n_heads}"
        )
    lines += ["", f"Modalities (--modality): {', '.join(MODALITIES)}"]
    lines += [f"Tasks (--task): {', '.join(TASKS)}"]
    return "\n".join(lines)


def parse_config(argv: Sequence[str] | None = None) -> ExperimentConfig:
    """Merge defaults <- YAML config <- command-line flags."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print(format_model_table())
        raise SystemExit(0)

    file_values: dict = {}
    if args.config:
        from .utils.io import load_yaml

        file_values = load_yaml(args.config)
    base = ExperimentConfig.from_dict(file_values)
    values = base.to_dict()
    values.pop("window_duration_ms", None)

    skip = {"list_models", "config"}
    for key, value in vars(args).items():
        if key in skip or value is None:
            continue
        values[key] = value

    # An explicit --window-size on the CLI should beat a --window-ms from YAML.
    if args.window_size is not None and args.window_ms is None:
        values["window_ms"] = None
    if values.get("cache_dir") == "":
        values["cache_dir"] = None
    # Auto-derive the run name unless it was set on the CLI or in the config file.
    if args.run_name is None and "run_name" not in file_values:
        values["run_name"] = None

    return ExperimentConfig.from_dict(values)
