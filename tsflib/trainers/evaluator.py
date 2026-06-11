"""
TSFLib Evaluator Implementation

This module provides evaluation utilities for time series forecasting models.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
from lightning.pytorch import Trainer

from tsflib.metrics import MetricsCalculator


class TSEvaluator:
    """Evaluator for time series forecasting models.

    This class provides methods for evaluating trained models on test data
    and computing various metrics.

    Args:
        trainer: PyTorch Lightning Trainer instance (optional).
        accelerator: Accelerator type if creating new trainer.
        devices: Number of devices if creating new trainer.

    Example:
        >>> evaluator = TSEvaluator()
        >>> metrics = evaluator.evaluate(model, datamodule)
        >>> print(metrics)
        >>>
        >>> # Or with existing trainer
        >>> evaluator = TSEvaluator(trainer=trainer)
        >>> results = evaluator.test(model, datamodule)
    """

    def __init__(
        self,
        trainer: Optional[Trainer] = None,
        accelerator: str = "auto",
        devices: Union[int, str] = "auto",
    ):
        self.trainer = trainer or Trainer(
            accelerator=accelerator,
            devices=devices,
            enable_progress_bar=True,
        )
        self.metrics_calculator = MetricsCalculator()

    def evaluate(
        self,
        model: L.LightningModule,
        datamodule: L.LightningDataModule,
        ckpt_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate a model on test data.

        Args:
            model: Model to evaluate.
            datamodule: Data module with test data.
            ckpt_path: Path to checkpoint to load.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Run test
        results = self.trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            verbose=True,
        )

        # Extract metrics from results
        metrics = results[0] if results else {}

        return metrics

    def test(
        self,
        model: L.LightningModule,
        datamodule: L.LightningDataModule,
        ckpt_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run test on a model.

        Args:
            model: Model to test.
            datamodule: Data module with test data.
            ckpt_path: Path to checkpoint to load.

        Returns:
            Test results.
        """
        return self.trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            verbose=True,
        )

    def predict(
        self,
        model: L.LightningModule,
        datamodule: L.LightningDataModule,
        ckpt_path: Optional[str] = None,
        return_predictions: bool = True,
    ) -> Any:
        """Generate predictions from a model.

        Args:
            model: Model to use for prediction.
            datamodule: Data module with data.
            ckpt_path: Path to checkpoint to load.
            return_predictions: Whether to return predictions.

        Returns:
            Predictions.
        """
        return self.trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            return_predictions=return_predictions,
        )

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            predictions: Predicted values (N, C, H) or (N, H).
            targets: Ground truth values (N, C, H) or (N, H).
            prefix: Prefix for metric names.

        Returns:
            Dictionary of metrics.
        """
        return self.metrics_calculator.compute_all(
            predictions, targets, prefix=prefix
        )

    def cross_validate(
        self,
        model_class: type,
        datamodule: L.LightningDataModule,
        n_splits: int = 5,
        **model_kwargs
    ) -> Dict[str, List[float]]:
        """Perform cross-validation.

        Note: This is a placeholder for future implementation.
        Cross-validation for time series requires special handling
        to avoid data leakage.

        Args:
            model_class: Model class to instantiate.
            datamodule: Data module.
            n_splits: Number of splits.
            **model_kwargs: Arguments for model initialization.

        Returns:
            Dictionary of metrics for each fold.
        """
        raise NotImplementedError(
            "Cross-validation for time series will be implemented in a future version. "
            "Time series CV requires special handling to prevent data leakage."
        )


class BatchEvaluator:
    """Evaluator for batch predictions.

    This class is useful for evaluating predictions outside of
    the Lightning training loop.

    Example:
        >>> evaluator = BatchEvaluator()
        >>>
        >>> model.eval()
        >>> with torch.no_grad():
        ...     for batch in dataloader:
        ...         x, y = batch[0], batch[1]
        ...         pred = model(x)
        ...         evaluator.add_batch(pred, y)
        >>>
        >>> metrics = evaluator.compute()
        >>> print(metrics)
    """

    def __init__(self):
        self.predictions: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.metrics_calculator = MetricsCalculator()

    def add_batch(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Add a batch of predictions and targets.

        Args:
            predictions: Predicted values.
            targets: Ground truth values.
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.predictions.append(predictions)
        self.targets.append(targets)

    def compute(self) -> Dict[str, float]:
        """Compute metrics for all accumulated batches.

        Returns:
            Dictionary of metrics.
        """
        if not self.predictions:
            return {}

        all_preds = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)

        return self.metrics_calculator.compute_all(all_preds, all_targets)

    def reset(self) -> None:
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.targets = []

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all accumulated predictions and targets.

        Returns:
            Tuple of (predictions, targets).
        """
        if not self.predictions:
            return np.array([]), np.array([])

        all_preds = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)

        return all_preds, all_targets


class ModelComparator:
    """Compare multiple models on the same dataset.

    Example:
        >>> comparator = ModelComparator()
        >>>
        >>> # Add models
        >>> comparator.add_model("iTransformer", model1)
        >>> comparator.add_model("DLinear", model2)
        >>>
        >>> # Compare
        >>> results = comparator.compare(datamodule)
        >>> print(results.to_dataframe())
    """

    def __init__(self):
        self.models: Dict[str, L.LightningModule] = {}
        self.results: Dict[str, Dict[str, float]] = {}

    def add_model(
        self,
        name: str,
        model: L.LightningModule,
        ckpt_path: Optional[str] = None,
    ) -> None:
        """Add a model to compare.

        Args:
            name: Model name.
            model: Model instance.
            ckpt_path: Optional checkpoint path to load.
        """
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])

        self.models[name] = model

    def compare(
        self,
        datamodule: L.LightningDataModule,
        metrics: Optional[List[str]] = None,
    ) -> "ComparisonResults":
        """Compare all models.

        Args:
            datamodule: Data module with test data.
            metrics: List of metrics to compute (None for all).

        Returns:
            ComparisonResults object.
        """
        evaluator = TSEvaluator()

        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            metrics_dict = evaluator.evaluate(model, datamodule)
            self.results[name] = metrics_dict

        return ComparisonResults(self.results)


class ComparisonResults:
    """Results from model comparison."""

    def __init__(self, results: Dict[str, Dict[str, float]]):
        self.results = results

    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.results).T

    def get_best(self, metric: str = "MSE", mode: str = "min") -> str:
        """Get the best model for a metric.

        Args:
            metric: Metric name.
            mode: 'min' or 'max'.

        Returns:
            Name of the best model.
        """
        values = {
            name: metrics.get(metric, float("inf") if mode == "min" else float("-inf"))
            for name, metrics in self.results.items()
        }

        if mode == "min":
            return min(values, key=values.get)
        else:
            return max(values, key=values.get)

    def __repr__(self) -> str:
        return f"ComparisonResults(models={list(self.results.keys())})"


__all__ = [
    "TSEvaluator",
    "BatchEvaluator",
    "ModelComparator",
    "ComparisonResults",
]
