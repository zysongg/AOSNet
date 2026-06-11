"""
TSFLib Metrics Module

This module provides evaluation metrics for time series forecasting.

Example:
    >>> from tsflib.metrics import MetricsCalculator, mse, mae
    >>>
    >>> calculator = MetricsCalculator()
    >>> metrics = calculator.compute_all(predictions, targets)
    >>>
    >>> # Or compute individual metrics
    >>> error = mse(predictions, targets)
"""

from tsflib.metrics.metrics import (
    MetricsCalculator,
    mse,
    mae,
    rmse,
    mape,
    smape,
    mse_per_feature,
    mae_per_feature,
    r2_score,
    correlation,
    POINT_METRICS,
    PROBABILISTIC_METRICS,
    compute_point_metrics,
    compute_prob_metrics,
    mse_torch,
    mae_torch,
    rmse_torch,
    mape_torch,
    smape_torch,
)

__all__ = [
    "MetricsCalculator",
    "mse",
    "mae",
    "rmse",
    "mape",
    "smape",
    "mse_per_feature",
    "mae_per_feature",
    "r2_score",
    "correlation",
    "POINT_METRICS",
    "PROBABILISTIC_METRICS",
    "compute_point_metrics",
    "compute_prob_metrics",
    "mse_torch",
    "mae_torch",
    "rmse_torch",
    "mape_torch",
    "smape_torch",
]
