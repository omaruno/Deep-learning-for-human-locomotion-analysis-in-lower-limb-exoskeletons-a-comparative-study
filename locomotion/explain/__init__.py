"""Explainable AI: SHAP-based sensor importance."""

from .shap_analysis import (
    channel_importance,
    compute_shap_values,
    group_importance,
    plot_channel_importance,
    plot_group_importance,
    run_shap_analysis,
)

__all__ = [
    "channel_importance",
    "compute_shap_values",
    "group_importance",
    "plot_channel_importance",
    "plot_group_importance",
    "run_shap_analysis",
]
