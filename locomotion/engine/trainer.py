"""Training / evaluation loop shared by every model and task."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .early_stopping import EarlyStopping
from .metrics import compute_metrics

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Per-epoch losses, for plotting and for the run report."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = 0
    stopped_early: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
        }


class Trainer:
    """Trains one model on one fold and evaluates it on the held-out split."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        regression: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 100,
        patience: int = 10,
        grad_clip: float = 1.0,
        class_weights: torch.Tensor | None = None,
        verbose: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.regression = regression
        self.epochs = int(epochs)
        self.grad_clip = float(grad_clip)
        self.verbose = verbose

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        if regression:
            self.criterion: nn.Module = nn.MSELoss()
        else:
            weights = class_weights.to(device) if class_weights is not None else None
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.early_stopping = EarlyStopping(patience=patience, mode="min")

    # ------------------------------------------------------------------ loops #
    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        total_items = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                if train:
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                total_loss += float(loss.item()) * inputs.size(0)
                total_items += inputs.size(0)
        return total_loss / max(total_items, 1)

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainingHistory:
        """Train with early stopping on the validation loss."""
        history = TrainingHistory()
        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            history.train_loss.append(train_loss)

            # With no validation split, early stopping monitors the train loss.
            monitored = train_loss
            if val_loader is not None:
                monitored = self._run_epoch(val_loader, train=False)
                history.val_loss.append(monitored)

            improved = self.early_stopping.step(monitored, self.model, epoch)
            history.epochs_run = epoch
            if self.verbose:
                LOGGER.info(
                    "epoch %3d/%d  train_loss=%.5f%s%s",
                    epoch,
                    self.epochs,
                    train_loss,
                    f"  val_loss={monitored:.5f}" if val_loader is not None else "",
                    "  *" if improved else "",
                )
            if self.early_stopping.should_stop:
                history.stopped_early = True
                LOGGER.info(
                    "Early stopping at epoch %d (best epoch %d)",
                    epoch,
                    self.early_stopping.best_epoch,
                )
                break

        self.early_stopping.restore(self.model)
        history.best_epoch = self.early_stopping.best_epoch
        return history

    # ------------------------------------------------------------ evaluation #
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(y_true, y_pred)`` over ``loader``."""
        self.model.eval()
        predictions: List[np.ndarray] = []
        truths: List[np.ndarray] = []
        for inputs, targets in loader:
            outputs = self.model(inputs.to(self.device, non_blocking=True))
            if self.regression:
                predictions.append(outputs.squeeze(-1).cpu().numpy())
                truths.append(targets.squeeze(-1).numpy())
            else:
                predictions.append(outputs.argmax(dim=1).cpu().numpy())
                truths.append(targets.numpy())
        return np.concatenate(truths), np.concatenate(predictions)

    def evaluate(self, loader: DataLoader) -> Dict[str, object]:
        """Metrics for the selected task on ``loader``."""
        y_true, y_pred = self.predict(loader)
        metrics = compute_metrics(y_true, y_pred, self.regression)
        metrics["loss"] = self._run_epoch(loader, train=False)
        return metrics

    @torch.no_grad()
    def measure_latency(
        self, loader: DataLoader, n_batches: int = 20, warmup: int = 3
    ) -> Dict[str, float]:
        """Single-window inference latency, as reported in the paper.

        Timing uses batch size 1 on the trainer's device, which is what an
        exoskeleton controller would actually see.
        """
        self.model.eval()
        sample = next(iter(loader))[0][:1].to(self.device)
        for _ in range(warmup):
            self.model(sample)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        timings: List[float] = []
        for _ in range(n_batches):
            start = time.perf_counter()
            self.model(sample)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
        array = np.asarray(timings)
        return {
            "latency_ms_mean": float(array.mean()),
            "latency_ms_std": float(array.std(ddof=0)),
            "latency_ms_p95": float(np.percentile(array, 95)),
        }
