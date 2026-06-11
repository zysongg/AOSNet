"""
Base Model Class for TSFLib

This module defines the abstract base class that all time series forecasting models
must inherit from. It provides a common interface for model operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class ModelBase(nn.Module, ABC):
    """Abstract base class for all time series forecasting models.

    All models in TSFLib should inherit from this class and implement
    the required abstract methods.

    Args:
        configs: Configuration object containing model hyperparameters.
            Required fields:
            - num_features: Number of input features
            - lookback_len: Length of input sequence
            - horizon_len: Length of forecast horizon
            - forecast_type: Type of forecasting ('M', 'MS', 'S', 'SS')

    Example:
        >>> class MyModel(ModelBase):
        ...     def __init__(self, configs):
        ...         super().__init__(configs)
        ...         self.encoder = nn.Linear(configs.lookback_len, configs.horizon_len)
        ...
        ...     def forward(self, batch_x):
        ...         return self.encoder(batch_x)
        ...
        ...     def train_loss(self, batch_):
        ...         pred = self.forward(batch_[0])
        ...         return F.mse_loss(pred, batch_[1])
    """

    def __init__(self, configs: Any):
        super().__init__()
        self.configs = configs

        # Extract common configuration
        self.num_features = getattr(configs, "num_features", None)
        self.lookback_len = getattr(configs, "lookback_len", None)
        self.horizon_len = getattr(configs, "horizon_len", None)
        self.forecast_type = getattr(configs, "forecast_type", "M")
        self.label_len = getattr(configs, "label_len", 0)

        # Store device for model operations
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self) -> torch.device:
        """Get the device where model parameters are stored."""
        return next(self.parameters()).device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, num_features, lookback_len)
                or a batch tuple from data loader.

        Returns:
            Output tensor of shape (batch_size, num_features, horizon_len)
        """
        pass

    @abstractmethod
    def train_loss(self, batch_: Tuple) -> torch.Tensor:
        """Compute training loss for a batch.

        Args:
            batch_: Tuple containing (batch_x, batch_y, batch_x_mark, batch_y_mark, index)
                - batch_x: Input data (B, C, L)
                - batch_y: Target data (B, C, H)
                - batch_x_mark: Time features for input (B, C, L)
                - batch_y_mark: Time features for target (B, C, H)
                - index: Sample indices (B,)

        Returns:
            Scalar loss tensor.
        """
        pass

    def val_loss(self, batch_: Tuple) -> torch.Tensor:
        """Compute validation loss for a batch.

        Default implementation uses train_loss. Override if different
        validation loss is needed.

        Args:
            batch_: Batch tuple from data loader.

        Returns:
            Scalar loss tensor.
        """
        return self.train_loss(batch_)

    def test_loss(self, batch_: Tuple) -> torch.Tensor:
        """Compute test loss for a batch.

        Default implementation uses val_loss. Override if different
        test loss is needed.

        Args:
            batch_: Batch tuple from data loader.

        Returns:
            Scalar loss tensor.
        """
        return self.val_loss(batch_)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Generate predictions for input data.

        This is a convenience method that calls forward() in eval mode.

        Args:
            x: Input tensor of shape (batch_size, num_features, lookback_len)

        Returns:
            Predictions of shape (batch_size, num_features, horizon_len)

        Example:
            >>> model.eval()
            >>> with torch.no_grad():
            ...     predictions = model.predict(test_data)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_num_params(self) -> int:
        """Get the total number of parameters in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """Get model size information.

        Returns:
            Dictionary containing:
            - num_params: Total number of parameters
            - model_size_mb: Model size in megabytes
        """
        num_params = self.get_num_params()
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024

        return {
            "num_params": num_params,
            "model_size_mb": model_size_mb,
        }

    def save(self, path: str) -> None:
        """Save model state dict to file.

        Args:
            path: Path to save the model.

        Example:
            >>> model.save("checkpoints/my_model.pt")
        """
        torch.save({
            "state_dict": self.state_dict(),
            "configs": self.configs,
        }, path)

    def load(self, path: str, strict: bool = True) -> None:
        """Load model state dict from file.

        Args:
            path: Path to the saved model.
            strict: Whether to strictly enforce that all keys match.

        Example:
            >>> model.load("checkpoints/my_model.pt")
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"], strict=strict)


def identity_norm(x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
    """Identity normalization function (no-op).

    Args:
        x: Input tensor.
        mode: Normalization mode (unused).

    Returns:
        Input tensor unchanged.
    """
    return x
