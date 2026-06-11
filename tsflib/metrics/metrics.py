"""
Evaluation Metrics for Time Series Forecasting

This module implements common metrics used for evaluating time series
forecasting models.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Squared Error.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        MSE value.
    """
    return float(np.mean((predictions - targets) ** 2))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(predictions - targets)))


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(mse(predictions, targets)))


def mape(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        eps: Small value to avoid division by zero.

    Returns:
        MAPE value in percentage.
    """
    return float(np.mean(np.abs((targets - predictions) / (targets + eps))) * 100)


def smape(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        eps: Small value to avoid division by zero.

    Returns:
        SMAPE value in percentage.
    """
    return float(
        np.mean(
            2 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets) + eps)
        ) * 100
    )


def mse_per_feature(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """MSE per feature/channel.

    Args:
        predictions: Predicted values of shape (N, C, H) or (N, C).
        targets: Ground truth values of shape (N, C, H) or (N, C).

    Returns:
        MSE for each feature.
    """
    if predictions.ndim == 3:
        return np.mean((predictions - targets) ** 2, axis=(0, 2))
    return np.mean((predictions - targets) ** 2, axis=0)


def mae_per_feature(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """MAE per feature/channel.

    Args:
        predictions: Predicted values of shape (N, C, H) or (N, C).
        targets: Ground truth values of shape (N, C, H) or (N, C).

    Returns:
        MAE for each feature.
    """
    if predictions.ndim == 3:
        return np.mean(np.abs(predictions - targets), axis=(0, 2))
    return np.mean(np.abs(predictions - targets), axis=0)


def r2_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """R-squared (coefficient of determination).

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        R2 score.
    """
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return float(1 - (ss_res / (ss_tot + 1e-8)))


def correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Pearson correlation coefficient.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Correlation coefficient.
    """
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    pred_mean = np.mean(pred_flat)
    target_mean = np.mean(target_flat)

    pred_std = np.std(pred_flat)
    target_std = np.std(target_flat)

    covariance = np.mean((pred_flat - pred_mean) * (target_flat - target_mean))
    correlation = covariance / (pred_std * target_std + 1e-8)

    return float(correlation)


class MetricsCalculator:
    """Calculator for time series forecasting metrics.

    This class provides methods to compute various evaluation metrics
    for time series forecasting models.

    Example:
        >>> calculator = MetricsCalculator()
        >>> metrics = calculator.compute_all(predictions, targets)
        >>> print(metrics)
        {'MSE': 0.05, 'MAE': 0.18, 'RMSE': 0.22, ...}
    """

    def __init__(self):
        self.metrics = {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "SMAPE": smape,
            "R2": r2_score,
            "Correlation": correlation,
        }

    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute specified metrics.

        Args:
            predictions: Predicted values.
            targets: Ground truth values.
            metric_names: List of metric names to compute (None for all).

        Returns:
            Dictionary of metric names to values.
        """
        if metric_names is None:
            metric_names = list(self.metrics.keys())

        results = {}
        for name in metric_names:
            if name in self.metrics:
                try:
                    results[name] = self.metrics[name](predictions, targets)
                except Exception as e:
                    results[name] = float("nan")
                    print(f"Warning: Failed to compute {name}: {e}")

        return results

    def compute_all(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute all available metrics.

        Args:
            predictions: Predicted values.
            targets: Ground truth values.
            prefix: Prefix for metric names.

        Returns:
            Dictionary of metric names to values.
        """
        results = self.compute(predictions, targets)
        if prefix:
            results = {f"{prefix}{k}": v for k, v in results.items()}
        return results

    def compute_per_feature(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute metrics per feature.

        Args:
            predictions: Predicted values of shape (N, C, H).
            targets: Ground truth values of shape (N, C, H).

        Returns:
            Dictionary of metric names to per-feature values.
        """
        return {
            "MSE": mse_per_feature(predictions, targets),
            "MAE": mae_per_feature(predictions, targets),
        }

    def add_metric(self, name: str, func) -> None:
        """Add a custom metric.

        Args:
            name: Metric name.
            func: Metric function that takes (predictions, targets) as arguments.
        """
        self.metrics[name] = func

    def list_metrics(self) -> List[str]:
        """List available metrics.

        Returns:
            List of metric names.
        """
        return list(self.metrics.keys())


# PyTorch versions for use in training loops


def mse_torch(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE loss for PyTorch tensors."""
    return F.mse_loss(predictions, targets)


def mae_torch(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MAE loss for PyTorch tensors."""
    return F.l1_loss(predictions, targets)


def rmse_torch(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """RMSE for PyTorch tensors."""
    return torch.sqrt(F.mse_loss(predictions, targets))


def mape_torch(
    predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """MAPE for PyTorch tensors."""
    return torch.mean(torch.abs((targets - predictions) / (targets + eps))) * 100


def smape_torch(
    predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """SMAPE for PyTorch tensors."""
    return (
        torch.mean(
            2
            * torch.abs(predictions - targets)
            / (torch.abs(predictions) + torch.abs(targets) + eps)
        )
        * 100
    )


POINT_METRICS: List[str] = ["MSE", "RMSE", "MAE", "MAPE", "sMAPE", "ND"]
PROBABILISTIC_METRICS: List[str] = [
    "CRPS",
    "CRPS_sum",
    "PICP",
    "QICE",
    "ND",
    "MSE_Median",
    "MAE_Median",
]


def _select_metrics(
    requested: Optional[Sequence[str]],
    available: Sequence[str],
) -> List[str]:
    """Resolve requested metric names, preserving the default order."""
    if requested is None:
        return list(available)

    available_map = {name.lower(): name for name in available}
    selected = []
    unknown = []
    for name in requested:
        canonical = available_map.get(name.lower())
        if canonical is None:
            unknown.append(name)
        elif canonical not in selected:
            selected.append(canonical)

    if unknown:
        raise ValueError(
            f"Unknown metric(s): {unknown}. Available metrics: {list(available)}"
        )
    return selected


def compute_point_metrics(
    targets: np.ndarray,
    forecasts: np.ndarray,
    metric_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Compute selected point forecasting metrics.

    Args:
        targets: Ground truth array.
        forecasts: Point forecast array with the same shape as targets.
        metric_names: Optional metric names. If omitted, computes all point
            metrics in ``POINT_METRICS``.

    Returns:
        Dictionary of metric name to value.
    """
    selected = _select_metrics(metric_names, POINT_METRICS)
    results: Dict[str, float] = {}
    cache: Dict[str, Union[np.ndarray, float]] = {}

    def abs_error() -> np.ndarray:
        if "abs_error" not in cache:
            cache["abs_error"] = np.abs(targets - forecasts)
        return cache["abs_error"]  # type: ignore[return-value]

    def abs_targets() -> np.ndarray:
        if "abs_targets" not in cache:
            cache["abs_targets"] = np.abs(targets)
        return cache["abs_targets"]  # type: ignore[return-value]

    if "MSE" in selected or "RMSE" in selected:
        mse_value = float(np.mean(np.square(targets - forecasts)))
        if "MSE" in selected:
            results["MSE"] = mse_value
        if "RMSE" in selected:
            results["RMSE"] = float(np.sqrt(mse_value))

    if "MAE" in selected:
        results["MAE"] = float(np.mean(abs_error()))

    if "MAPE" in selected:
        target_abs = abs_targets()
        mask = target_abs > 1e-8
        results["MAPE"] = (
            float(np.mean(abs_error()[mask] / target_abs[mask])) if mask.any() else 0.0
        )

    if "sMAPE" in selected:
        denom = abs_targets() + np.abs(forecasts)
        mask = denom > 1e-8
        results["sMAPE"] = (
            float(2 * np.mean(abs_error()[mask] / denom[mask])) if mask.any() else 0.0
        )

    if "ND" in selected:
        abs_target_sum = float(np.sum(abs_targets()))
        results["ND"] = (
            float(np.sum(abs_error()) / abs_target_sum) if abs_target_sum > 0 else 0.0
        )

    return {name: results[name] for name in selected if name in results}


def _quantile_loss(targets: np.ndarray, forecasts: np.ndarray, q: float) -> np.ndarray:
    """Quantile loss element values."""
    return 2 * np.abs((forecasts - targets) * ((targets <= forecasts) - q))


def _weighted_quantile_loss(targets: np.ndarray, forecasts: np.ndarray, q: float) -> float:
    """ProbTS-compatible weighted quantile loss for one sequence."""
    abs_target_sum = float(np.sum(np.abs(targets)))
    if abs_target_sum <= 0:
        return 0.0
    return float(np.sum(_quantile_loss(targets, forecasts, q)) / abs_target_sum)


def _sequence_nd(targets: np.ndarray, forecasts: np.ndarray) -> float:
    """ProbTS-compatible ND for one sequence."""
    abs_target_sum = float(np.sum(np.abs(targets)))
    if abs_target_sum <= 0:
        return 0.0
    return float(np.sum(np.abs(targets - forecasts)) / abs_target_sum)


def _flatten_sample_last(samples: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten ``(N, S, H, C)`` samples to nsdiff ``(N * H * C, S)`` layout."""
    samples_flat = np.moveaxis(samples, 1, -1).reshape(-1, samples.shape[1])
    targets_flat = targets.reshape(-1)
    return samples_flat, targets_flat


def _compute_picp_from_flat(samples_flat: np.ndarray, targets_flat: np.ndarray) -> float:
    lower_bound, upper_bound = np.quantile(samples_flat, [0.05, 0.95], axis=1)
    in_range = (targets_flat >= lower_bound) & (targets_flat <= upper_bound)
    return float(np.mean(in_range))


def _compute_qice_from_flat(
    samples_flat: np.ndarray,
    targets_flat: np.ndarray,
    n_bins: int = 10,
) -> float:
    quantile_list = np.arange(n_bins + 1) * (100.0 / n_bins)
    y_pred_quantiles = np.percentile(samples_flat, q=quantile_list, axis=1)
    quantile_membership = ((targets_flat - y_pred_quantiles) > 0).astype(int)
    y_true_quantile_membership = quantile_membership.sum(axis=0)
    bin_counts = np.array(
        [(y_true_quantile_membership == v).sum() for v in range(n_bins + 2)]
    )
    bin_counts[1] += bin_counts[0]
    bin_counts[-2] += bin_counts[-1]
    bin_counts = bin_counts[1:-1]
    ratio_by_bin = bin_counts.astype(float) / samples_flat.shape[0]
    return float(np.mean(np.abs(np.ones(n_bins) / n_bins - ratio_by_bin)))


def compute_prob_metrics(
    targets: np.ndarray,
    samples: np.ndarray,
    metric_names: Optional[Sequence[str]] = None,
    quantiles_num: int = 10,
) -> Dict[str, float]:
    """Compute selected probabilistic forecasting metrics.

    Args:
        targets: Ground truth, shape ``(N, prediction_length, target_dim)``.
        samples: Forecast samples, shape ``(N, num_samples, prediction_length,
            target_dim)``.
        metric_names: Optional metric names. If omitted, computes all metrics in
            ``PROBABILISTIC_METRICS``.
        quantiles_num: Number of quantiles for CRPS computation.

    Returns:
        Dictionary of metric name to value.
    """
    selected = _select_metrics(metric_names, PROBABILISTIC_METRICS)

    # Detect and clean NaN/inf in inputs
    nan_count_samples = np.sum(~np.isfinite(samples))
    nan_count_targets = np.sum(~np.isfinite(targets))
    if nan_count_samples > 0:
        total = samples.size
        pct = nan_count_samples / total * 100
        print(
            f"Warning: {nan_count_samples}/{total} ({pct:.2f}%) non-finite values "
            f"in samples, replacing with nan_to_num"
        )
        samples = np.nan_to_num(samples, nan=0.0, posinf=1e6, neginf=-1e6)
    if nan_count_targets > 0:
        total = targets.size
        pct = nan_count_targets / total * 100
        print(
            f"Warning: {nan_count_targets}/{total} ({pct:.2f}%) non-finite values "
            f"in targets, replacing with nan_to_num"
        )
        targets = np.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)

    quantiles = (1.0 * np.arange(quantiles_num) / quantiles_num)[1:]
    results: Dict[str, float] = {}
    cache: Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = {}

    def median_forecasts() -> np.ndarray:
        if "median" not in cache:
            cache["median"] = np.quantile(samples, 0.5, axis=1)
        return cache["median"]  # type: ignore[return-value]

    def flat_samples_targets() -> Tuple[np.ndarray, np.ndarray]:
        if "flat" not in cache:
            cache["flat"] = _flatten_sample_last(samples, targets)
        return cache["flat"]  # type: ignore[return-value]

    if "MSE_Median" in selected:
        median = median_forecasts()
        results["MSE_Median"] = float(np.mean(np.square(targets - median)))

    if "MAE_Median" in selected:
        median = median_forecasts()
        results["MAE_Median"] = float(np.mean(np.abs(targets - median)))

    if "ND" in selected:
        median = median_forecasts()
        results["ND"] = float(
            np.mean([
                _sequence_nd(targets[i], median[i])
                for i in range(targets.shape[0])
            ])
        )

    if "CRPS" in selected:
        crps_by_sequence = []
        for i in range(targets.shape[0]):
            weighted_ql = []
            for q in quantiles:
                q_forecasts = np.quantile(samples[i], q, axis=0)
                weighted_ql.append(_weighted_quantile_loss(targets[i], q_forecasts, q))
            crps_by_sequence.append(np.mean(weighted_ql))
        results["CRPS"] = float(np.mean(crps_by_sequence))

    if "CRPS_sum" in selected:
        targets_sum = targets.sum(axis=-1)
        samples_sum = samples.sum(axis=-1)
        crps_sum_by_sequence = []
        for i in range(targets_sum.shape[0]):
            weighted_ql_sum = []
            for q in quantiles:
                q_forecasts_sum = np.quantile(samples_sum[i], q, axis=0)
                weighted_ql_sum.append(
                    _weighted_quantile_loss(targets_sum[i], q_forecasts_sum, q)
                )
            crps_sum_by_sequence.append(np.mean(weighted_ql_sum))
        results["CRPS_sum"] = float(np.mean(crps_sum_by_sequence))

    if "PICP" in selected:
        samples_flat, targets_flat = flat_samples_targets()
        results["PICP"] = _compute_picp_from_flat(samples_flat, targets_flat)

    if "QICE" in selected:
        samples_flat, targets_flat = flat_samples_targets()
        results["QICE"] = _compute_qice_from_flat(samples_flat, targets_flat)

    return {name: results[name] for name in selected if name in results}

__all__ = [
    # NumPy versions
    "mse",
    "mae",
    "rmse",
    "mape",
    "smape",
    "mse_per_feature",
    "mae_per_feature",
    "r2_score",
    "correlation",
    "MetricsCalculator",
    "POINT_METRICS",
    "PROBABILISTIC_METRICS",
    "compute_point_metrics",
    "compute_prob_metrics",
    # PyTorch versions
    "mse_torch",
    "mae_torch",
    "rmse_torch",
    "mape_torch",
    "smape_torch",
]
