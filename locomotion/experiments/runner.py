"""Experiment runner: builds the data, trains every fold, reports and explains."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ..config import ExperimentConfig
from ..data import LocomotionDataModule
from ..engine import Trainer, aggregate_folds, primary_metric_name
from ..models import build_model
from ..utils import resolve_device, save_json, save_yaml, set_seed

LOGGER = logging.getLogger(__name__)


def _fold_iterator(datamodule: LocomotionDataModule, config: ExperimentConfig):
    """Yield the folds implied by ``--evaluation``."""
    if config.evaluation == "holdout":
        yield datamodule.holdout_split(config.test_fraction)
        return
    if config.evaluation != "loso":
        raise ValueError(
            f"Unknown evaluation scheme '{config.evaluation}'; use 'loso' or 'holdout'"
        )
    for index, fold in enumerate(datamodule.loso_splits(config.subjects)):
        if config.max_folds is not None and index >= config.max_folds:
            LOGGER.info("Stopping after %d fold(s) (--max-folds)", config.max_folds)
            break
        yield fold


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    """Run one full experiment and write everything under ``config.run_dir``.

    Returns the summary dict that is also saved as ``results.json``.
    """
    started = time.time()
    set_seed(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config.to_dict(), run_dir / "config.yaml")

    LOGGER.info("=" * 78)
    LOGGER.info(
        "Run '%s' | model=%s (%s) | modality=%s | task=%s | window=%d samples (%.0f ms)",
        config.run_name,
        config.model,
        config.model_size,
        config.modality,
        config.task,
        config.window_size,
        config.window_duration_ms,
    )
    LOGGER.info("Device: %s | output: %s", device, run_dir)
    LOGGER.info("=" * 78)

    datamodule = LocomotionDataModule(
        data_dir=config.data_dir,
        modality=config.modality,
        window_size=config.window_size,
        stride=config.stride,
        task=config.task,
        imu_locations=config.imu_locations,
        emg_muscles=config.emg_muscles,
        data_layout=config.data_layout,
        file_pattern=config.file_pattern,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_fraction=config.val_fraction,
        cache_dir=config.cache_dir,
        seed=config.seed,
    ).setup()

    fold_metrics: List[Dict[str, object]] = []
    shap_reports: List[Dict[str, object]] = []
    n_parameters = 0

    for fold_index, (held_out, train_index, test_index) in enumerate(
        _fold_iterator(datamodule, config)
    ):
        LOGGER.info(
            "--- fold %d | held-out: %s | train=%d windows, test=%d windows ---",
            fold_index + 1,
            held_out,
            len(train_index),
            len(test_index),
        )
        # Re-seed per fold so folds are independent but reproducible.
        set_seed(config.seed + fold_index, deterministic=config.deterministic)
        train_loader, val_loader, test_loader, scaler = datamodule.dataloaders(
            train_index, test_index
        )

        model = build_model(
            config.model,
            c_in=datamodule.n_channels,
            c_out=datamodule.n_outputs,
            seq_len=config.window_size,
            size=config.model_size,
            dropout=config.dropout,
            **config.size_overrides(),
        )
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold_index == 0:
            LOGGER.info("Model '%s' has %s trainable parameters", config.model, f"{n_parameters:,}")

        weights = (
            datamodule.class_weights(train_index)
            if config.class_weights and not datamodule.is_regression
            else None
        )
        trainer = Trainer(
            model,
            device=device,
            regression=datamodule.is_regression,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            patience=config.patience,
            grad_clip=config.grad_clip,
            class_weights=weights,
        )
        history = trainer.fit(train_loader, val_loader)
        metrics = trainer.evaluate(test_loader)
        metrics["fold"] = fold_index
        metrics["held_out_subject"] = held_out
        metrics["n_train_windows"] = int(len(train_index))
        metrics["n_test_windows"] = int(len(test_index))
        metrics["epochs_run"] = history.epochs_run
        if config.measure_latency:
            metrics.update(trainer.measure_latency(test_loader))

        primary = primary_metric_name(datamodule.is_regression)
        LOGGER.info(
            "fold %d done | %s=%.4f | loss=%.5f",
            fold_index + 1,
            primary,
            float(metrics[primary]),
            float(metrics["loss"]),
        )

        fold_dir = run_dir / f"fold_{fold_index:02d}_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {"metrics": metrics, "history": history.to_dict(), "scaler": scaler.to_dict()},
            fold_dir / "fold.json",
        )
        if config.save_checkpoints:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "channels": datamodule.channels,
                    "scaler": scaler.to_dict(),
                },
                fold_dir / "model.pt",
            )

        if config.shap and fold_index < config.shap_folds:
            from ..explain import run_shap_analysis

            LOGGER.info("Running SHAP analysis for fold %d", fold_index + 1)
            report = run_shap_analysis(
                model,
                background=scaler.transform(datamodule.windows.x[train_index]),
                samples=scaler.transform(datamodule.windows.x[test_index]),
                channels=datamodule.channels,
                output_dir=fold_dir / "shap",
                device=device,
                regression=datamodule.is_regression,
                n_background=config.shap_background,
                n_samples=config.shap_samples,
                seed=config.seed,
            )
            report["fold"] = fold_index
            report["held_out_subject"] = held_out
            shap_reports.append(report)

        fold_metrics.append(metrics)

    if not fold_metrics:
        raise RuntimeError("No fold was evaluated; check --subjects / --evaluation")

    summary = aggregate_folds(fold_metrics)
    primary = primary_metric_name(datamodule.is_regression)
    results = {
        "run_name": config.run_name,
        "config": config.to_dict(),
        "n_parameters": n_parameters,
        "n_folds": len(fold_metrics),
        "channels": datamodule.channels,
        "primary_metric": primary,
        "summary": summary,
        "folds": fold_metrics,
        "shap": shap_reports,
        "runtime_seconds": round(time.time() - started, 1),
    }
    save_json(results, run_dir / "results.json")
    _save_fold_table(fold_metrics, run_dir / "folds.csv")
    if shap_reports:
        _save_shap_summary(shap_reports, run_dir / "shap_summary.csv")

    _log_summary(config, summary, primary, datamodule.target_unit)
    return results


def _save_fold_table(fold_metrics: List[Dict[str, object]], path: Path) -> None:
    """Flat per-fold CSV, easy to paste into a paper table."""
    rows = [
        {
            key: value
            for key, value in fold.items()
            if isinstance(value, (int, float, str))
        }
        for fold in fold_metrics
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_shap_summary(shap_reports: List[Dict[str, object]], path: Path) -> None:
    """Sensor-group importance per explained fold."""
    rows = []
    for report in shap_reports:
        row = {"fold": report["fold"], "held_out_subject": report["held_out_subject"]}
        row.update(report["sensor_group_importance"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _log_summary(
    config: ExperimentConfig, summary: dict, primary: str, unit: str
) -> None:
    LOGGER.info("=" * 78)
    LOGGER.info("RESULTS  %s", config.run_name)
    LOGGER.info("-" * 78)
    interesting = dict.fromkeys(
        [primary, "precision", "recall", "f1", "mae", "rmse", "r2", "latency_ms_mean"]
    )
    for key in interesting:
        if key in summary:
            stats = summary[key]
            suffix = f" {unit}" if unit and key in {"mae", "rmse"} else ""
            LOGGER.info(
                "%-16s %.4f +/- %.4f%s   (min %.4f, max %.4f, %d folds)",
                key,
                stats["mean"],
                stats["std"],
                suffix,
                stats["min"],
                stats["max"],
                stats["n_folds"],
            )
    LOGGER.info("=" * 78)
