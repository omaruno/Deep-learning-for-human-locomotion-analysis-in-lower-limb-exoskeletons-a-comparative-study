"""Evaluation metrics for the classification and regression tasks."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)

from ..data.constants import CLASS_DISPLAY_NAMES, NUM_CLASSES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """Accuracy, weighted/macro precision-recall-F1 and the confusion matrix."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = list(range(NUM_CLASSES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "per_class_f1": {
            name: float(value)
            for name, value in zip(
                CLASS_DISPLAY_NAMES,
                f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, MSE, RMSE and the coefficient of determination."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mse = float(mean_squared_error(y_true, y_pred))
    # R^2 is undefined when the held-out subject only walked one ramp/stair.
    r2 = float(r2_score(y_true, y_pred)) if np.ptp(y_true) > 0 else float("nan")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
    }


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, regression: bool
) -> Dict[str, object]:
    return regression_metrics(y_true, y_pred) if regression else classification_metrics(y_true, y_pred)


def primary_metric_name(regression: bool) -> str:
    """Metric used to rank models and to report ``mean +/- std`` over folds."""
    return "mae" if regression else "accuracy"


def aggregate_folds(fold_metrics: list) -> Dict[str, Dict[str, float]]:
    """Mean and standard deviation of every scalar metric across LOSO folds."""
    if not fold_metrics:
        return {}
    scalar_keys = [
        key
        for key in fold_metrics[0]
        if isinstance(fold_metrics[0][key], (int, float))
    ]
    summary: Dict[str, Dict[str, float]] = {}
    for key in scalar_keys:
        values = np.asarray(
            [fold[key] for fold in fold_metrics if key in fold], dtype=np.float64
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "n_folds": int(values.size),
        }
    return summary
