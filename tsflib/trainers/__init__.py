"""
TSFLib Trainers Module

This module provides training and evaluation utilities for time series
forecasting models using PyTorch Lightning.

Example:
    >>> from tsflib.trainers import TSTrainer, TSEvaluator
    >>>
    >>> # Create trainer
    >>> trainer = TSTrainer(
    ...     max_epochs=100,
    ...     accelerator='gpu',
    ...     devices=1
    ... )
    >>>
    >>> # Train model
    >>> trainer.fit(model, datamodule=datamodule)
    >>>
    >>> # Evaluate
    >>> evaluator = TSEvaluator(trainer)
    >>> metrics = evaluator.evaluate(model, datamodule)
"""

from tsflib.trainers.trainer import TSTrainer, TSTrainerConfig, TSTrainerBuilder
from tsflib.trainers.evaluator import TSEvaluator, BatchEvaluator, ModelComparator
from tsflib.trainers.callbacks import (
    MetricsCallback,
    PredictionWriter,
    TrainingTimer,
    EarlyStoppingWithWarmup,
    GradientMonitor,
)

__all__ = [
    # Trainer
    "TSTrainer",
    "TSTrainerConfig",
    "TSTrainerBuilder",
    # Evaluator
    "TSEvaluator",
    "BatchEvaluator",
    "ModelComparator",
    # Callbacks
    "MetricsCallback",
    "PredictionWriter",
    "TrainingTimer",
    "EarlyStoppingWithWarmup",
    "GradientMonitor",
]
