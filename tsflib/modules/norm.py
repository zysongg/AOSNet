"""
Normalization Modules for Time Series Forecasting

This module provides various normalization techniques commonly used in
time series forecasting models.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization.

    RevIN normalizes each instance (sample) independently across the time dimension,
    making it suitable for non-stationary time series. The normalization is reversible,
    allowing the model to learn on normalized data while predictions can be
    denormalized back to the original scale.

    Reference: Kim et al., "Reversible Instance Normalization for Accurate
    Time-Series Forecasting against Distribution Shift", ICLR 2022.

    Args:
        num_features: Number of features or channels.
        eps: Small value for numerical stability.
        affine: If True, apply learnable affine transformation.
        subtract_last: If True, subtract last value instead of mean.
        time_dim: Index of the time dimension. Defaults to -1 (last dim).
        feature_dim: Index of the feature dimension. Defaults to 1.

    Example:
        >>> norm = RevIN(num_features=7, affine=True)
        >>> x = torch.randn(32, 7, 96)  # (batch, features, time)
        >>> x_norm = norm(x, mode='norm')
        >>> x_denorm = norm(x_norm, mode='denorm')
    """

    def __init__(
        self,
        use_norm: bool,
        num_features: int,
        time_dim: int = -1,
        feature_dim: int = 1,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
    ):
        super().__init__()

        self.use_norm = use_norm
        if self.use_norm:
            self.num_features = num_features
            self.eps = eps
            self.affine = affine
            self.subtract_last = subtract_last
            self.time_dim = time_dim
            self.feature_dim = feature_dim

            # Learnable affine parameters
            if self.affine:
                self.affine_weight = nn.Parameter(torch.ones(num_features))
                self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """Apply normalization or denormalization.

        Args:
            x: Input tensor of shape (batch, ..., time, ..., feature, ...),
                with dimensions configured by ``time_dim`` and ``feature_dim``.
            mode: 'norm' for normalization, 'denorm' for denormalization.

        Returns:
            Normalized or denormalized tensor.
        """
        if not self.use_norm:
            return x  # Identity if normalization is disabled
        if mode == "norm":
            return self.normalize(x)
        elif mode == "denorm":
            return self.denormalize(x)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'norm' or 'denorm'.")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor.

        Args:
            x: Input tensor with layout configured by ``time_dim`` and ``feature_dim``.

        Returns:
            Normalized tensor.
        """
        self._validate_input(x)

        # Compute statistics along time dimension
        if self.subtract_last:
            self.last = x.narrow(self.time_dim, x.size(self.time_dim) - 1, 1).detach()
        else:
            self.mean = torch.mean(x, dim=self.time_dim, keepdim=True).detach()

        self.stdev = torch.sqrt(
            torch.var(x, dim=self.time_dim, keepdim=True, unbiased=False) + self.eps
        ).detach()

        # Normalize
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev

        # Apply affine transformation
        if self.affine:
            x = x * self._reshape_affine(self.affine_weight, x)
            x = x + self._reshape_affine(self.affine_bias, x)

        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input tensor.

        Args:
            x: Normalized tensor with layout configured by ``time_dim`` and ``feature_dim``.

        Returns:
            Denormalized tensor.
        """
        self._validate_input(x)
        if not hasattr(self, "stdev") or self.stdev is None:
            raise RuntimeError("RevIN must be called with mode='norm' before mode='denorm'.")

        # Remove affine transformation
        if self.affine:
            x = x - self._reshape_affine(self.affine_bias, x)
            x = x / self._reshape_affine(self.affine_weight + self.eps * self.eps, x)

        # Denormalize
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean

        return x

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor dimension and feature size.

        Checks that the input is at least 3D and that the feature dimension
        matches the expected ``num_features``.

        Args:
            x: Input tensor to validate.

        Raises:
            ValueError: If input dimensionality or feature size is incorrect.
        """
        if x.dim() < 3:
            raise ValueError(f"RevIN expects at least 3D input, got {x.dim()}D.")
        feature_dim = self.feature_dim if self.feature_dim >= 0 else x.dim() + self.feature_dim
        if x.size(feature_dim) != self.num_features:
            time_dim = self.time_dim if self.time_dim >= 0 else x.dim() + self.time_dim
            raise ValueError(
                f"Expected {self.num_features} features on dim {feature_dim} "
                f"(time_dim={self.time_dim}, feature_dim={self.feature_dim}), "
                f"got shape {tuple(x.shape)}."
            )

    def _reshape_affine(self, value: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape 1D affine parameters to match the input tensor shape for broadcasting.

        The affine weight and bias are 1D tensors of shape ``(num_features,)``.
        This method reshapes them to a tensor with the same number of dimensions
        as ``x``, setting the feature dimension to ``num_features`` and all other
        dimensions to 1, so they can broadcast correctly.

        Example:
            If ``value`` has shape ``(7,)`` and ``x`` has shape ``(32, 7, 96)``
            with ``feature_dim=1``, the result will have shape ``(1, 7, 1)``.

        Args:
            value: 1D tensor of affine parameters (weight or bias).
            x: Reference input tensor whose dimensionality is used for reshaping.

        Returns:
            Reshaped tensor ready for broadcasting with ``x``.
        """
        shape = [1] * x.dim()
        feature_dim = self.feature_dim if self.feature_dim >= 0 else x.dim() + self.feature_dim
        shape[feature_dim] = self.num_features
        return value.view(*shape)


class MinMaxNorm(nn.Module):
    """Min-Max normalization to [0, 1] range.

    Computes per-feature min/max statistics over the batch and time dimensions,
    then scales each feature to [0, 1].

    Args:
        num_features: Number of features.
        eps: Small value for numerical stability.
        time_dim: Index of the time dimension. Defaults to -1 (last dim).
        feature_dim: Index of the feature dimension. Defaults to 1.

    Example:
        >>> norm = MinMaxNorm(num_features=7)
        >>> norm.fit(train_data)
        >>> x_norm = norm(x, mode='norm')
        >>> x_denorm = norm(x_norm, mode='denorm')
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        time_dim: int = -1,
        feature_dim: int = 1,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.time_dim = time_dim
        self.feature_dim = feature_dim

        self.register_buffer("min", torch.zeros(num_features))
        self.register_buffer("max", torch.ones(num_features))
        self.register_buffer("fitted", torch.tensor(False))

    def _resolve_dim(self, dim: int, ndim: int) -> int:
        """Resolve a possibly-negative dimension index to a positive index."""
        return dim if dim >= 0 else ndim + dim

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor dimension and feature size."""
        if x.dim() < 2:
            raise ValueError(f"MinMaxNorm expects at least 2D input, got {x.dim()}D.")
        feature_dim = self._resolve_dim(self.feature_dim, x.dim())
        if x.size(feature_dim) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features on dim {feature_dim} "
                f"(time_dim={self.time_dim}, feature_dim={self.feature_dim}), "
                f"got shape {tuple(x.shape)}."
            )

    def _reshape_stat(self, value: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape 1D statistics to match input tensor shape for broadcasting."""
        shape = [1] * x.dim()
        feature_dim = self._resolve_dim(self.feature_dim, x.dim())
        shape[feature_dim] = self.num_features
        return value.view(*shape)

    def fit(self, x: torch.Tensor) -> None:
        """Compute min/max from training data.

        Statistics are computed over all dimensions except the feature dimension.

        Args:
            x: Training data of shape (batch, ..., time, ..., feature, ...).
        """
        self._validate_input(x)
        feature_dim = self._resolve_dim(self.feature_dim, x.dim())
        # Compute stats over all dims except feature_dim
        reduce_dims = [d for d in range(x.dim()) if d != feature_dim]
        # Flatten reduce dims, compute min/max
        if len(reduce_dims) == 1:
            self.min = x.min(dim=reduce_dims[0])[0]
            self.max = x.max(dim=reduce_dims[0])[0]
        else:
            # Use flatten approach for multiple reduce dims
            self.min = torch.amin(x, dim=reduce_dims)
            self.max = torch.amax(x, dim=reduce_dims)

        self.fitted = torch.tensor(True)

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """Apply min-max normalization or denormalization."""
        self._validate_input(x)
        if not self.fitted and mode == "norm":
            raise RuntimeError("MinMaxNorm must be fitted before normalization.")

        min_val = self._reshape_stat(self.min, x)
        max_val = self._reshape_stat(self.max, x)
        range_val = max_val - min_val + self.eps

        if mode == "norm":
            return (x - min_val) / range_val
        elif mode == "denorm":
            return x * range_val + min_val
        else:
            raise ValueError(f"Invalid mode: {mode}")


# Registry for easy access
NORM_REGISTRY = {
    "revin": RevIN,
    "minmax": MinMaxNorm,
}


def get_norm(name: str, **kwargs):
    """Get a normalization module by name.

    Args:
        name: Normalization name.
        **kwargs: Additional arguments.

    Returns:
        Normalization module.
    """
    if name not in NORM_REGISTRY:
        available = ", ".join(NORM_REGISTRY.keys())
        raise ValueError(f"Norm '{name}' not found. Available: {available}")
    return NORM_REGISTRY[name](**kwargs)


__all__ = [
    "RevIN",
    "MinMaxNorm",
    "get_norm",
    "NORM_REGISTRY",
]
