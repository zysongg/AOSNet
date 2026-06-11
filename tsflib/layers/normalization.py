"""
Normalization Layers for Time Series Forecasting

This module provides various normalization layers used in time series models.
"""

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """Normalization layer with optional affine parameters.

    Supports both mean subtraction and last-value subtraction modes.

    Args:
        num_features: Number of features or channels
        eps: Value added for numerical stability
        affine: If True, use learnable affine parameters
        subtract_last: If True, subtract last value instead of mean
        non_norm: If True, bypass normalization

    Example:
        >>> norm = Normalize(num_features=7, affine=True)
        >>> x_norm = norm(x, mode='norm')
        >>> x_denorm = norm(x_norm, mode='denorm')
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
        non_norm: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm

        if self.affine:
            self._init_params()

    def _init_params(self):
        """Initialize affine parameters."""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """Compute normalization statistics."""
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        """Apply normalization."""
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """Apply denormalization."""
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor
            mode: 'norm' or 'denorm'

        Returns:
            Normalized or denormalized tensor
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode {mode} not supported")
        return x


__all__ = ["Normalize"]
