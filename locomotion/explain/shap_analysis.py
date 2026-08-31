"""SHAP feature-importance analysis for the trained models.

The analysis answers the question the paper asks of it: *which sensors
actually matter?* It attributes each prediction to the input channels, then
aggregates the absolute attributions over samples and over the time axis to
rank channels and sensor groups (foot / shank / thigh / trunk / EMG).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..data.constants import (
    CLASS_DISPLAY_NAMES,
    IMU_LOCATIONS,
    channel_group,
    short_channel_names,
)

LOGGER = logging.getLogger(__name__)

#: Colour per sensor group, reused across every SHAP figure.
GROUP_COLORS: Dict[str, str] = {
    "foot": "#2ca02c",
    "shank": "#d62728",
    "thigh": "#1f77b4",
    "trunk": "#ff7f0e",
    "emg": "#9467bd",
}


def _normalise_shap_output(shap_values, n_outputs: int) -> List[np.ndarray]:
    """Return one ``(n_samples, n_channels, window)`` array per model output.

    ``shap`` has changed this return type across versions: older releases give
    a list (one entry per output), newer ones stack the outputs on a trailing
    axis. Both are handled here.
    """
    if isinstance(shap_values, list):
        return [np.asarray(value) for value in shap_values]
    values = np.asarray(shap_values)
    if values.ndim == 4 and values.shape[-1] == n_outputs:
        return [values[..., index] for index in range(n_outputs)]
    return [values]


def compute_shap_values(
    model: nn.Module,
    background: np.ndarray,
    samples: np.ndarray,
    device: torch.device,
    n_background: int = 100,
    n_samples: int = 200,
    seed: int = 42,
) -> List[np.ndarray]:
    """Run ``shap.GradientExplainer`` on a trained model.

    Args:
        model: trained network, consuming ``(B, C, T)`` tensors.
        background: training windows used as the reference distribution.
        samples: windows to explain (typically the held-out subject).
        n_background: how many background windows to keep (SHAP cost is
            linear in this number).
        n_samples: how many windows to explain.

    Returns:
        One array of shape ``(n_samples, n_channels, window)`` per model output.
    """
    try:
        import shap
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "SHAP analysis requires the 'shap' package: pip install shap"
        ) from error

    rng = np.random.default_rng(seed)
    background_index = rng.choice(
        len(background), size=min(n_background, len(background)), replace=False
    )
    sample_index = rng.choice(
        len(samples), size=min(n_samples, len(samples)), replace=False
    )

    model = model.to(device).eval()
    background_tensor = torch.from_numpy(
        np.ascontiguousarray(background[background_index], dtype=np.float32)
    ).to(device)
    sample_tensor = torch.from_numpy(
        np.ascontiguousarray(samples[sample_index], dtype=np.float32)
    ).to(device)

    with torch.no_grad():
        n_outputs = int(model(sample_tensor[:1]).shape[1])

    LOGGER.info(
        "Computing SHAP values: %d background / %d explained windows, %d output(s)",
        len(background_index),
        len(sample_index),
        n_outputs,
    )
    explainer = shap.GradientExplainer(model, background_tensor)
    values = explainer.shap_values(sample_tensor)
    return _normalise_shap_output(values, n_outputs)


def channel_importance(shap_values: Sequence[np.ndarray]) -> np.ndarray:
    """Mean absolute SHAP value per channel, averaged over outputs and time.

    Returns a ``(n_channels,)`` array. Averaging over the whole window (rather
    than reading a single time step, as the original notebook did) makes the
    ranking far less sensitive to where inside the window one looks.
    """
    per_output = [np.abs(np.asarray(value)).mean(axis=(0, 2)) for value in shap_values]
    return np.mean(np.stack(per_output, axis=0), axis=0)


def per_class_importance(shap_values: Sequence[np.ndarray]) -> np.ndarray:
    """``(n_outputs, n_channels)`` mean absolute SHAP values."""
    return np.stack(
        [np.abs(np.asarray(value)).mean(axis=(0, 2)) for value in shap_values], axis=0
    )


def group_importance(
    importance: np.ndarray, channels: Sequence[str]
) -> Dict[str, float]:
    """Average importance per sensor group, i.e. per physical device."""
    groups: Dict[str, List[float]] = {}
    for channel, value in zip(channels, importance):
        groups.setdefault(channel_group(channel), []).append(float(value))
    return {name: float(np.mean(values)) for name, values in groups.items()}


def plot_channel_importance(
    importance: np.ndarray,
    channels: Sequence[str],
    output_path,
    title: str = "Absolute mean SHAP value per channel",
):
    """Bar chart coloured by sensor group, with a dashed mean line per group."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = short_channel_names(channels)
    groups = [channel_group(channel) for channel in channels]
    colors = [GROUP_COLORS.get(group, "#7f7f7f") for group in groups]

    figure, axes = plt.subplots(figsize=(max(10, len(channels) * 0.55), 6))
    axes.bar(names, importance, color=colors)
    axes.set_xlabel("Channel")
    axes.set_ylabel("Absolute mean SHAP value")
    axes.set_title(title)
    axes.tick_params(axis="x", rotation=90)

    for group in dict.fromkeys(groups):
        mask = np.asarray([g == group for g in groups])
        axes.axhline(
            float(importance[mask].mean()),
            color=GROUP_COLORS.get(group, "#7f7f7f"),
            linestyle="--",
            linewidth=1.2,
            label=f"mean {group}",
        )
    axes.legend(loc="upper right", fontsize=8)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def plot_group_importance(groups: Dict[str, float], output_path, title: str = "Sensor importance"):
    """Bar chart of the per-sensor-group averages."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [name for name in (*IMU_LOCATIONS, "emg") if name in groups]
    values = [groups[name] for name in order]
    colors = [GROUP_COLORS.get(name, "#7f7f7f") for name in order]

    figure, axes = plt.subplots(figsize=(6, 4))
    axes.bar(order, values, color=colors)
    axes.set_ylabel("Absolute mean SHAP value")
    axes.set_title(title)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def run_shap_analysis(
    model: nn.Module,
    background: np.ndarray,
    samples: np.ndarray,
    channels: Sequence[str],
    output_dir,
    device: torch.device,
    regression: bool = False,
    n_background: int = 100,
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[str, object]:
    """Full SHAP pipeline: values -> rankings -> CSV + figures.

    Returns a summary dict (also written to ``shap_importance.json``) with the
    ranked channels and the per-sensor-group averages.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_values = compute_shap_values(
        model,
        background,
        samples,
        device=device,
        n_background=n_background,
        n_samples=n_samples,
        seed=seed,
    )
    importance = channel_importance(shap_values)
    per_class = per_class_importance(shap_values)
    groups = group_importance(importance, channels)

    table = pd.DataFrame(
        {
            "channel": list(channels),
            "sensor_group": [channel_group(channel) for channel in channels],
            "mean_abs_shap": importance,
        }
    )
    if not regression and per_class.shape[0] == len(CLASS_DISPLAY_NAMES):
        for index, class_name in enumerate(CLASS_DISPLAY_NAMES):
            table[f"shap_{class_name}"] = per_class[index]
    table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    csv_path = output_dir / "shap_channel_importance.csv"
    table.to_csv(csv_path, index=False)

    figures = [
        str(
            plot_channel_importance(
                importance, channels, output_dir / "shap_channel_importance.png"
            )
        ),
        str(
            plot_group_importance(
                groups, output_dir / "shap_sensor_importance.png"
            )
        ),
    ]
    np.save(output_dir / "shap_values.npy", np.stack(shap_values, axis=0))

    ranking = table[["channel", "sensor_group", "mean_abs_shap"]].to_dict("records")
    LOGGER.info(
        "SHAP: most informative channel is '%s' (%.5f); sensor ranking: %s",
        ranking[0]["channel"],
        ranking[0]["mean_abs_shap"],
        " > ".join(
            name for name, _ in sorted(groups.items(), key=lambda kv: -kv[1])
        ),
    )
    return {
        "channel_ranking": ranking,
        "sensor_group_importance": groups,
        "most_important_channel": ranking[0]["channel"],
        "least_important_sensor_group": min(groups, key=groups.get),
        "csv": str(csv_path),
        "figures": figures,
    }
