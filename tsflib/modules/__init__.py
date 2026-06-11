"""
TSFLib Modules

This module provides various neural network modules for time series forecasting,
including normalization layers, embeddings, and utility modules.

Example:
    >>> from tsflib.modules import RevIN
    >>>
    >>> norm = RevIN(num_features=7, affine=True)
    >>> normalized = norm(x, mode='norm')
    >>> denormalized = norm(normalized, mode='denorm')
"""

from tsflib.modules.norm import MinMaxNorm, RevIN
from tsflib.modules.embed import (
    DataEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    PositionalEmbedding,
)
from tsflib.modules.patching import PatchingLayer
from tsflib.modules.TimeToken import TimeToken

__all__ = [
    # Normalization
    "RevIN",
    "MinMaxNorm",
    # Embeddings
    "DataEmbedding",
    "TemporalEmbedding",
    "TimeFeatureEmbedding",
    "PositionalEmbedding",
    # Tokenization
    "PatchingLayer",
    "TimeToken",
]
