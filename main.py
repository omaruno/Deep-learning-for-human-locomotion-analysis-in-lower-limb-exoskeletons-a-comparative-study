#!/usr/bin/env python
"""Single entry point for every experiment in this repository.

Pick a model, a sensor modality, a temporal window and a model size; optionally
run the SHAP sensor-importance analysis afterwards.

    python main.py --model lstm --modality imu --window-size 100
    python main.py --model tst --modality multimodal --model-size large --shap
    python main.py --model cnn_lstm --task stair_height --evaluation loso
    python main.py --list-models

Run ``python main.py --help`` for the full list of options.
"""

from __future__ import annotations

import sys

from locomotion.cli import parse_config
from locomotion.experiments import run_experiment
from locomotion.utils import setup_logging


def main(argv=None) -> int:
    config = parse_config(argv)
    setup_logging(config.log_level, log_file=config.run_dir / "run.log")
    try:
        run_experiment(config)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        # Configuration and data problems should read as a message, not a stack trace.
        print(f"\nError: {error}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
