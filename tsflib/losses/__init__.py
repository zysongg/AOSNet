"""
TSFLib Loss Functions Module

This module provides various loss functions for time series forecasting tasks.

Example:
    >>> from tsflib.losses import MSELoss, MAELoss
    >>>
    >>> criterion = MSELoss()
    >>> loss = criterion(pred, target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union, Tuple


def _get_loss_class(name: str):
    """Get a loss class by name from the registry."""
    if name not in LOSS_REGISTRY:
        available = ", ".join(LOSS_REGISTRY.keys())
        raise ValueError(f"Loss '{name}' not found. Available: {available}")
    return LOSS_REGISTRY[name]


class MSELoss(nn.Module):
    """Mean Squared Error Loss for time series forecasting.

    Args:
        weight: Optional weight tensor for feature-wise weighting. Set to None
            when used as a standalone loss (weighting is for multi-loss scenarios).

    Example:
        >>> criterion = MSELoss()
        >>> loss = criterion(pred, target)
    """

    def __init__(self, weight: Optional[List[float]] = None):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            pred: Predictions of shape (B, C, L) or (B, L).
            target: Ground truth of shape (B, C, L) or (B, L).

        Returns:
            Scalar loss value.
        """
        return F.mse_loss(pred, target)


class MAELoss(nn.Module):
    """Mean Absolute Error Loss for time series forecasting.

    Args:
        weight: Optional weight tensor for feature-wise weighting. Set to None
            when used as a standalone loss (weighting is for multi-loss scenarios).

    Example:
        >>> criterion = MAELoss()
        >>> loss = criterion(pred, target)
    """

    def __init__(self, weight: Optional[List[float]] = None):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MAE loss.

        Args:
            pred: Predictions of shape (B, C, L) or (B, L).
            target: Ground truth of shape (B, C, L) or (B, L).

        Returns:
            Scalar loss value.
        """
        return F.l1_loss(pred, target)


class MixFreqMAELoss(nn.Module):
    """Mixed Frequency MAE Loss.

    Combines MAE in both time and frequency domains.

    Args:
        weight: Optional weight tensor for feature-wise weighting (unused for
            single-loss scenarios, kept for API consistency).
        config: Alpha value for frequency domain weight. ``config[0]`` controls
            the trade-off: ``total = (1 - alpha) * time_loss + alpha * freq_loss``.
            Defaults to ``[0.5]``.

    Example:
        >>> criterion = MixFreqMAELoss(config=[0.3])
        >>> loss = criterion(pred, target)
    """

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        config: List[float] = [0.5],
    ):
        super().__init__()
        self.alpha = config[0]
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight))
        else:
            self.weight = None
        self.mae_loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mixed frequency MAE loss.

        Args:
            pred: Predictions of shape (B, C, L).
            target: Ground truth of shape (B, C, L).

        Returns:
            Scalar loss value.
        """
        time_domain_loss = self.mae_loss(pred, target)
        freq_domain_loss = self.mae_loss(
            torch.fft.rfft(pred, dim=-1), torch.fft.rfft(target, dim=-1)
        )
        total_loss = (1 - self.alpha) * time_domain_loss + self.alpha * freq_domain_loss
        return total_loss


class MixFreqMSELoss(nn.Module):
    """Mixed Frequency MSE Loss.

    Combines MSE in the time domain and MAE in the frequency domain.

    Args:
        weight: Optional weight tensor for feature-wise weighting (unused for
            single-loss scenarios, kept for API consistency).
        config: Alpha value for frequency domain weight. ``config[0]`` controls
            the trade-off: ``total = (1 - alpha) * time_loss + alpha * freq_loss``.
            Defaults to ``[0.5]``.

    Example:
        >>> criterion = MixFreqMSELoss(config=[0.3])
        >>> loss = criterion(pred, target)
    """

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        config: List[float] = [0.5],
    ):
        super().__init__()
        self.alpha = config[0]
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight))
        else:
            self.weight = None
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mixed frequency MSE loss.

        Args:
            pred: Predictions of shape (B, C, L) or (B, L, C).
            target: Ground truth of shape (B, C, L) or (B, L, C).

        Returns:
            Scalar loss value.
        """
        time_domain_loss = self.mse_loss(pred, target)
        freq_domain_loss = self.mae_loss(
            torch.fft.rfft(pred, dim=-1), torch.fft.rfft(target, dim=-1)
        )
        total_loss = (1 - self.alpha) * time_domain_loss + self.alpha * freq_domain_loss
        return total_loss


class HuberLoss(nn.Module):
    """Huber Loss for robust time series forecasting.

    Huber loss is less sensitive to outliers than MSE.

    Args:
        weight: Optional weight tensor for feature-wise weighting (unused for
            single-loss scenarios, kept for API consistency).
        delta: Threshold for switching between L1 and L2 loss.

    Example:
        >>> criterion = HuberLoss(delta=1.0)
        >>> loss = criterion(pred, target)
    """

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        delta: float = 1.0,
    ):
        super().__init__()
        self.delta = delta
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight))
        else:
            self.weight = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss."""
        error = pred - target
        abs_error = torch.abs(error)

        quadratic = torch.min(abs_error, torch.tensor(self.delta, device=abs_error.device))
        linear = abs_error - quadratic

        return torch.mean(0.5 * quadratic**2 + self.delta * linear)


class QuantileLoss(nn.Module):
    """Quantile Loss for probabilistic forecasting.

    Args:
        weight: Optional weight tensor for feature-wise weighting (unused for
            single-loss scenarios, kept for API consistency).
        quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).

    Example:
        >>> criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        >>> loss = criterion(pred, target)  # pred shape: (B, C, L, num_quantiles)
    """

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ):
        super().__init__()
        self.quantiles = quantiles
        self.register_buffer("quantile_tensor", torch.tensor(quantiles))
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight))
        else:
            self.weight = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss.

        Args:
            pred: Predictions of shape (B, C, L, num_quantiles).
            target: Ground truth of shape (B, C, L).

        Returns:
            Scalar loss value.
        """
        # Expand target to match pred shape
        target = target.unsqueeze(-1)  # (B, C, L, 1)

        errors = target - pred
        losses = torch.max(
            (self.quantile_tensor - 1) * errors,
            self.quantile_tensor * errors,
        )

        return torch.mean(losses)


class TweedieLoss(nn.Module):
    """Tweedie Loss for zero-inflated data.

    Useful for forecasting sparse time series.

    Args:
        weight: Optional weight tensor for feature-wise weighting (unused for
            single-loss scenarios, kept for API consistency).
        p: Tweedie variance power (1 < p < 2).
    """

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        p: float = 1.5,
    ):
        super().__init__()
        assert 1 < p < 2, "p must be between 1 and 2"
        self.p = p
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight))
        else:
            self.weight = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Tweedie loss."""
        # Avoid log(0) by adding small epsilon
        eps = 1e-8
        pred = torch.clamp(pred, min=eps)

        a = target * torch.pow(pred, 1 - self.p) / (1 - self.p)
        b = torch.pow(pred, 2 - self.p) / (2 - self.p)

        return torch.mean(-a + b)


class DBLoss(nn.Module):
    """Decomposition-Based Loss.

    Decomposes predictions and targets into seasonal and trend components,
    then computes separate losses on each. The trend loss is dynamically
    scaled by the seasonal-to-trend loss ratio to balance optimization.

    Args:
        weight: Optional weight tensor for feature-wise weighting (unused for
            single-loss scenarios, kept for API consistency).
        config: ``[alpha, beta]`` where ``alpha`` is the EMA smoothing factor
            for decomposition, and ``beta`` weights the seasonal vs trend loss.
            ``total = beta * season_loss + (1 - beta) * trend_loss``.
            Defaults to ``[0.5, 0.5]``.

    Example:
        >>> criterion = DBLoss(config=[0.5, 0.5])
        >>> loss = criterion(pred, target)
    """

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        config: List[float] = [0.5, 0.5],
    ):
        super().__init__()
        from tsflib.layers.decomposition import DECOMP

        alpha, beta = config
        self.decomp = DECOMP("ema", alpha=alpha)
        self.beta = beta
        self.mse = nn.MSELoss(reduction="mean")
        self.mae = nn.L1Loss(reduction="mean")
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight))
        else:
            self.weight = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute decomposition-based loss.

        Args:
            pred: Predictions of shape (B, C, L).
            target: Ground truth of shape (B, C, L).

        Returns:
            Scalar loss value.
        """
        pred_season, pred_trend = self.decomp(pred)
        target_season, target_trend = self.decomp(target)

        season_loss = self.mse(pred_season, target_season)
        trend_loss = self.mae(pred_trend, target_trend)
        trend_loss = trend_loss * (season_loss / (trend_loss + 1e-8)).detach()

        return self.beta * season_loss + (1 - self.beta) * trend_loss


class LossManager(nn.Module):
    """Unified loss container that dispatches to specific loss functions.

    Supports both single-loss and multi-loss weighted combinations.

    Args:
        loss_type: Loss specification. Can be:
            - ``str``: Single loss name, e.g. ``"mse"``
            - ``List[str]``: ``[loss_name, alpha]`` where alpha is a config
              value passed as ``config=[alpha]`` to the loss constructor
            - ``List[str]``: ``[loss_name]`` single loss with no config
            - ``List[List]``: List of ``[loss_name, weight]`` or
              ``[loss_name, weight, config]`` for multi-loss
        loss_weights: Weights for each loss in multi-loss mode. If ``None``,
            all losses are weighted equally (1.0).

    Examples:
        Single loss:
            >>> mgr = LossManager("mse")
            >>> loss = mgr(pred, target)

        Single loss with config:
            >>> mgr = LossManager(["mix_freq_mse", 0.3])
            >>> loss = mgr(pred, target)

        Multiple losses with weights:
            >>> mgr = LossManager([["mse", 1.0], ["mae", 0.5]])
            >>> loss = mgr(pred, target)

        Multiple losses with weights and configs:
            >>> mgr = LossManager([
            ...     ["mix_freq_mse", 1.0, [0.3]],
            ...     ["db", 0.5, [0.5, 0.5]],
            ... ])
            >>> loss = mgr(pred, target)
    """

    def __init__(
        self,
        loss_type: Union[str, List],
        loss_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        # Normalize input to a list of (loss_class, weight, kwargs)
        self._losses: nn.ModuleList = nn.ModuleList()
        self._weights: List[float] = []

        # Case 1: single string -> single loss with weight 1.0
        if isinstance(loss_type, str):
            loss_cls = _get_loss_class(loss_type)
            self._losses.append(loss_cls())
            self._weights.append(1.0)
            return

        # Case 2: list — distinguish single-loss vs multi-loss
        if len(loss_type) == 0:
            raise ValueError("loss_type cannot be empty")

        first = loss_type[0]

        # Case 2a: [loss_name] or [loss_name, alpha]
        if isinstance(first, str):
            loss_name = first[0] if isinstance(first, list) else first
            loss_cls = _get_loss_class(loss_type[0] if isinstance(loss_type[0], str) else loss_type[0][0])

            if len(loss_type) == 1:
                # Single loss, no config
                self._losses.append(loss_cls())
            elif len(loss_type) == 2:
                # Single loss with config value: [name, alpha]
                self._losses.append(loss_cls(config=[loss_type[1]]))
            else:
                # [name, alpha, ...extra_kwargs] — treat remaining as kwargs
                extra = loss_type[2:]
                kwargs = {}
                for i, v in enumerate(extra):
                    kwargs[f"_arg{i}"] = v
                self._losses.append(loss_cls(config=[loss_type[1]], **kwargs))
            self._weights.append(1.0)
            return

        # Case 2b: [[name, weight], [name, weight, config], ...] — multi-loss
        for item in loss_type:
            name = item[0]
            weight = item[1] if len(item) > 1 else 1.0
            config = item[2] if len(item) > 2 else None

            loss_cls = _get_loss_class(name)
            if config is not None:
                self._losses.append(loss_cls(config=config))
            else:
                self._losses.append(loss_cls())
            self._weights.append(weight)

        # Normalize weights to sum to 1
        total = sum(self._weights)
        if total > 0:
            self._weights = [w / total for w in self._weights]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of all registered losses.

        Args:
            pred: Predictions.
            target: Ground truth.

        Returns:
            Scalar loss value (weighted sum).
        """
        if len(self._losses) == 1:
            return self._losses[0](pred, target)

        total_loss = torch.tensor(0.0, device=pred.device)
        for loss_fn, w in zip(self._losses, self._weights):
            total_loss = total_loss + w * loss_fn(pred, target)
        return total_loss

    def get_individual_losses(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute each loss separately (for logging).

        Args:
            pred: Predictions.
            target: Ground truth.

        Returns:
            Dict mapping loss index to individual loss value.
        """
        return {f"loss_{i}": fn(pred, target) for i, fn in enumerate(self._losses)}


# Loss registry for easy access
LOSS_REGISTRY = {
    "mse": MSELoss,
    "mae": MAELoss,
    "mix_freq_mse": MixFreqMSELoss,
    "mix_freq_mae": MixFreqMAELoss,
    "huber": HuberLoss,
    "quantile": QuantileLoss,
    "tweedie": TweedieLoss,
    "db": DBLoss,
}


def get_loss(name: str, **kwargs) -> nn.Module:
    """Get a loss function by name.

    Args:
        name: Loss function name.
        **kwargs: Additional arguments for the loss function.

    Returns:
        Loss function instance.

    Example:
        >>> criterion = get_loss("mse")
        >>> criterion = get_loss("huber", delta=0.5)
    """
    if name not in LOSS_REGISTRY:
        available = ", ".join(LOSS_REGISTRY.keys())
        raise ValueError(f"Loss '{name}' not found. Available: {available}")
    return LOSS_REGISTRY[name](**kwargs)


def list_losses() -> List[str]:
    """List all available loss functions.

    Returns:
        List of loss function names.
    """
    return list(LOSS_REGISTRY.keys())


__all__ = [
    "MSELoss",
    "MAELoss",
    "MixFreqMSELoss",
    "MixFreqMAELoss",
    "HuberLoss",
    "QuantileLoss",
    "TweedieLoss",
    "DBLoss",
    "LossManager",
    "get_loss",
    "list_losses",
    "LOSS_REGISTRY",
]
