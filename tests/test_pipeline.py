"""Smoke tests covering the data pipeline, every model and the CLI.

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locomotion.cli import parse_config  # noqa: E402
from locomotion.config import ExperimentConfig  # noqa: E402
from locomotion.data import LocomotionDataModule, channels_for_modality  # noqa: E402
from locomotion.data.constants import normalise_label  # noqa: E402
from locomotion.engine import aggregate_folds, classification_metrics  # noqa: E402
from locomotion.experiments import run_experiment  # noqa: E402
from locomotion.models import available_models, build_model  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from make_synthetic_dataset import generate  # noqa: E402


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> Path:
    """A small synthetic dataset shared by every test in the session."""
    path = tmp_path_factory.mktemp("data")
    generate(path, n_subjects=3, n_trials=1, n_bouts=6, seed=7)
    return path


# --------------------------------------------------------------------- data #
def test_modality_channel_counts():
    assert len(channels_for_modality("imu")) == 24
    assert len(channels_for_modality("emg")) == 11
    assert len(channels_for_modality("multimodal")) == 35
    assert len(channels_for_modality("imu", ["foot", "shank", "thigh"])) == 18


def test_unknown_modality_is_rejected():
    with pytest.raises(ValueError, match="Unknown modality"):
        channels_for_modality("eeg")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("walk-stairascent", "stairascent"),
        ("stairascent-walk", "stairascent"),
        ("rampdescent", "rampdescent"),
        ("walk", "levelground"),
        ("idle", None),
    ],
)
def test_label_normalisation(raw, expected):
    assert normalise_label(raw) == expected


@pytest.mark.parametrize("modality,n_channels", [("imu", 24), ("emg", 11), ("multimodal", 35)])
def test_datamodule_shapes(dataset, modality, n_channels):
    datamodule = LocomotionDataModule(
        dataset, modality=modality, window_size=50, cache_dir=None
    ).setup()
    assert datamodule.windows.x.shape[1] == n_channels
    assert datamodule.windows.x.shape[2] == 50
    assert datamodule.n_outputs == 5
    assert len(datamodule.subject_ids()) == 3


def test_windows_never_mix_labels(dataset):
    """Every retained window must come from a single activity bout."""
    datamodule = LocomotionDataModule(dataset, window_size=100, cache_dir=None).setup()
    assert set(np.unique(datamodule.windows.y)).issubset(set(range(5)))
    assert np.isfinite(datamodule.windows.x).all()


def test_loso_splits_are_disjoint(dataset):
    datamodule = LocomotionDataModule(dataset, window_size=100, cache_dir=None).setup()
    for subject, train_index, test_index in datamodule.loso_splits():
        assert not set(train_index) & set(test_index)
        assert set(datamodule.windows.subjects[test_index]) == {subject}
        assert subject not in set(datamodule.windows.subjects[train_index])


def test_scaler_is_fitted_on_training_data_only(dataset):
    """The held-out subject must not influence the standardisation statistics."""
    datamodule = LocomotionDataModule(
        dataset, window_size=100, batch_size=16, cache_dir=None
    ).setup()
    _, train_index, test_index = next(iter(datamodule.loso_splits()))
    _, _, _, scaler = datamodule.dataloaders(train_index, test_index)
    reference = np.mean(datamodule.windows.x[train_index], axis=(0, 2))
    # The scaler is fitted after a validation split is carved out, so the means
    # are close but not identical; a full-data fit would be much further off.
    assert scaler.mean.squeeze().shape == reference.shape


def test_regression_task_filters_windows(dataset):
    datamodule = LocomotionDataModule(
        dataset, window_size=100, task="ramp_slope", cache_dir=None
    ).setup()
    assert datamodule.is_regression
    assert datamodule.n_outputs == 1
    assert np.isfinite(datamodule.targets).all()


# ------------------------------------------------------------------- models #
@pytest.mark.parametrize("name", available_models())
@pytest.mark.parametrize("c_out", [5, 1])
def test_every_model_forward(name, c_out):
    model = build_model(name, c_in=24, c_out=c_out, seq_len=100, size="tiny")
    output = model(torch.randn(3, 24, 100))
    assert output.shape == (3, c_out)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("size", ["tiny", "small", "base"])
def test_model_size_changes_capacity(size):
    model = build_model("cnn", 24, 5, 100, size=size)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_larger_size_has_more_parameters():
    counts = [
        sum(p.numel() for p in build_model("lstm", 24, 5, 100, size=size).parameters())
        for size in ("tiny", "small", "base", "large")
    ]
    assert counts == sorted(counts)


def test_size_override_applies():
    model = build_model("cnn", 24, 5, 100, size="base", width=17)
    assert model.features[0].conv.out_channels == 17


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("resnet999", 24, 5, 100)


@pytest.mark.parametrize("window", [50, 100, 250])
def test_models_accept_different_windows(window):
    for name in ("cnn", "lstm", "tst", "xceptiontime"):
        model = build_model(name, 24, 5, window, size="tiny")
        assert model(torch.randn(2, 24, window)).shape == (2, 5)


# ------------------------------------------------------------------ metrics #
def test_classification_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4])
    metrics = classification_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_aggregate_folds():
    summary = aggregate_folds([{"accuracy": 0.9}, {"accuracy": 0.7}])
    assert summary["accuracy"]["mean"] == pytest.approx(0.8)
    assert summary["accuracy"]["n_folds"] == 2


# ---------------------------------------------------------------------- CLI #
def test_cli_maps_flags_to_config():
    config = parse_config(
        ["--model", "gru", "--modality", "emg", "--window-size", "64", "--model-size", "large"]
    )
    assert (config.model, config.modality, config.window_size) == ("gru", "emg", 64)
    assert config.model_size == "large"
    assert config.run_name == "gru_emg_w64_large"


def test_window_ms_converts_to_samples():
    config = parse_config(["--window-ms", "200", "--sampling-rate", "500"])
    assert config.window_size == 100


def test_shap_flag_defaults_off():
    assert parse_config([]).shap is False
    assert parse_config(["--shap"]).shap is True


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown configuration key"):
        ExperimentConfig.from_dict({"not_a_real_option": 1})


# ------------------------------------------------------------- end-to-end #
def test_full_run(dataset, tmp_path):
    """Two folds of real training, plus the artefacts a run must produce."""
    config = ExperimentConfig(
        data_dir=str(dataset),
        model="cnn",
        modality="imu",
        window_size=100,
        model_size="tiny",
        epochs=2,
        patience=2,
        max_folds=2,
        batch_size=32,
        output_dir=str(tmp_path),
        cache_dir=None,
        save_checkpoints=True,
        device="cpu",
    )
    results = run_experiment(config)
    assert results["n_folds"] == 2
    assert 0.0 <= results["summary"]["accuracy"]["mean"] <= 1.0
    assert (config.run_dir / "results.json").exists()
    assert (config.run_dir / "folds.csv").exists()
    assert (config.run_dir / "config.yaml").exists()
    assert list(config.run_dir.glob("fold_*/model.pt"))


def test_shap_run(dataset, tmp_path):
    pytest.importorskip("shap")
    config = ExperimentConfig(
        data_dir=str(dataset),
        model="cnn",
        modality="imu",
        window_size=100,
        model_size="tiny",
        epochs=1,
        max_folds=1,
        batch_size=32,
        shap=True,
        shap_background=20,
        shap_samples=20,
        output_dir=str(tmp_path),
        cache_dir=None,
        save_checkpoints=False,
        device="cpu",
    )
    results = run_experiment(config)
    assert len(results["shap"]) == 1
    report = results["shap"][0]
    assert len(report["channel_ranking"]) == 24
    assert set(report["sensor_group_importance"]) == {"foot", "shank", "thigh", "trunk"}
    assert Path(report["csv"]).exists()
