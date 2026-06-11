"""
TSFLib Model Wrappers

This module provides PyTorch Lightning wrappers for time series forecasting models,
allowing them to be used with PyTorch Lightning's training infrastructure.

Example:
    >>> from tsflib.wrappers import TSForecastingModule
    >>> from tsflib.models import iTransformer
    >>>
    >>> # Create model
    >>> model = iTransformer.Model(configs)
    >>>
    >>> # Wrap for Lightning
    >>> pl_module = TSForecastingModule(
    ...     model=model,
    ...     lr=1e-3,
    ...     loss='mse'
    ... )
    >>>
    >>> # Train with Lightning
    >>> trainer.fit(pl_module, datamodule)
"""

from tsflib.wrappers.lightning_module import (
    TSForecastingModule,
    TSForecastingConfig,
    TSProbabilisticModule,
)

__all__ = [
    "TSForecastingModule",
    "TSForecastingConfig",
    "TSProbabilisticModule",
]
