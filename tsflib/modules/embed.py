"""
Embedding Modules for Time Series Forecasting

This module provides various embedding layers for encoding temporal information
and feature representations in time series models.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding.

    Args:
        d_model: Dimension of the model.
        max_len: Maximum sequence length.

    Example:
        >>> embed = PositionalEmbedding(d_model=512, max_len=5000)
        >>> pos_enc = embed(x)  # x shape doesn't matter, returns (1, max_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Compute positional encodings
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get positional encoding.

        Args:
            x: Input tensor (used to determine batch size and device).

        Returns:
            Positional encoding of shape (1, seq_len, d_model).
        """
        return self.pe[:, : x.size(-1), :]


class TokenEmbedding(nn.Module):
    """1D convolutional token embedding.

    Args:
        c_in: Number of input channels.
        d_model: Dimension of the model.
        kernel_size: Kernel size for convolution.
        padding: Padding size.

    Example:
        >>> embed = TokenEmbedding(c_in=7, d_model=512)
        >>> x_embed = embed(x)  # x: (B, C, L) -> (B, d_model, L)
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        kernel_size: int = 3,
        padding: Optional[int] = None,
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2

        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply token embedding.

        Args:
            x: Input tensor of shape (B, C, L).

        Returns:
            Embedded tensor of shape (B, d_model, L).
        """
        x = self.tokenConv(x)
        return x.transpose(1, 2)  # (B, L, d_model)


class TemporalEmbedding(nn.Module):
    """Temporal embedding using month, day, weekday, hour, minute.

    Args:
        d_model: Dimension of the model.
        embed_type: Type of embedding ('fixed' or 'learned').

    Example:
        >>> embed = TemporalEmbedding(d_model=512)
        >>> x_mark = torch.randn(32, 4)  # (B, 4) - month, day, weekday, hour
        >>> temp_embed = embed(x_mark)  # (B, d_model)
    """

    def __init__(self, d_model: int, embed_type: str = "fixed"):
        super().__init__()

        minute_size = 4
        hour_size = 25
        weekday_size = 8
        day_size = 32
        month_size = 13

        # Embedding layers
        self.minute_embed = nn.Embedding(minute_size, d_model)
        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.month_embed = nn.Embedding(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal embedding.

        Args:
            x: Temporal features of shape (B, num_features).
                Expected order: [month, day, weekday, hour, minute]

        Returns:
            Temporal embedding of shape (B, d_model).
        """
        x = x.long()

        minute_x = self.minute_embed(x[:, 4]) if x.size(-1) > 4 else 0
        hour_x = self.hour_embed(x[:, 3])
        weekday_x = self.weekday_embed(x[:, 2])
        day_x = self.day_embed(x[:, 1])
        month_x = self.month_embed(x[:, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Time feature embedding using linear projection.

    Args:
        d_model: Dimension of the model.
        num_features: Number of time features.

    Example:
        >>> embed = TimeFeatureEmbedding(d_model=512, num_features=4)
        >>> x_mark = torch.randn(32, 96, 4)  # (B, L, 4)
        >>> time_embed = embed(x_mark)  # (B, L, d_model)
    """

    def __init__(self, d_model: int, num_features: int = 4):
        super().__init__()
        self.embed = nn.Linear(num_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time feature embedding.

        Args:
            x: Time features of shape (B, L, num_features).

        Returns:
            Embedded tensor of shape (B, L, d_model).
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """Combined data embedding with value, position, and temporal embeddings.

    Args:
        c_in: Number of input channels.
        d_model: Dimension of the model.
        embed_type: Type of temporal embedding ('timeF' or 'temporal').
        freq: Time frequency ('h', 't', 's', etc.).
        dropout: Dropout rate.

    Example:
        >>> embed = DataEmbedding(c_in=7, d_model=512, embed_type='timeF', freq='h')
        >>> x = torch.randn(32, 96, 7)  # (B, L, C)
        >>> x_mark = torch.randn(32, 96, 4)  # (B, L, num_time_features)
        >>> embedded = embed(x, x_mark)  # (B, L, d_model)
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "timeF",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        if embed_type == "timeF":
            # Determine num_features based on frequency
            freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
            num_features = freq_map.get(freq, 4)
            self.temporal_embedding = TimeFeatureEmbedding(d_model, num_features)
        elif embed_type == "temporal":
            self.temporal_embedding = TemporalEmbedding(d_model)
        else:
            self.temporal_embedding = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply data embedding.

        Args:
            x: Input data of shape (B, L, C).
            x_mark: Optional temporal features of shape (B, L, num_time_features).

        Returns:
            Embedded tensor of shape (B, L, d_model).
        """
        # Value embedding
        x = self.value_embedding(x.transpose(1, 2))  # (B, L, d_model)

        # Add positional encoding
        x = x + self.position_embedding(x)

        # Add temporal embedding if available
        if self.temporal_embedding is not None and x_mark is not None:
            x = x + self.temporal_embedding(x_mark)

        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """Patch embedding for time series (used in PatchTST).

    Args:
        seq_len: Sequence length.
        patch_len: Length of each patch.
        stride: Stride for patch extraction.
        d_model: Dimension of the model.
        padding: Padding strategy ('end' or None).
        dropout: Dropout rate.

    Example:
        >>> embed = PatchEmbedding(seq_len=96, patch_len=16, stride=8, d_model=512)
        >>> x = torch.randn(32, 7, 96)  # (B, C, L)
        >>> patches = embed(x)  # (B, num_patches, d_model)
    """

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        padding: Optional[str] = "end",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride

        # Calculate number of patches
        if padding == "end":
            # Pad at the end to ensure complete patches
            num_patches = (seq_len - patch_len) // stride + 2
            self.padding = nn.ReplicationPad1d((0, stride))
        else:
            num_patches = (seq_len - patch_len) // stride + 1
            self.padding = None

        self.num_patches = num_patches

        # Projection layer
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = PositionalEmbedding(d_model, max_len=num_patches)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply patch embedding.

        Args:
            x: Input tensor of shape (B, C, L).

        Returns:
            Patch embeddings of shape (B * C, num_patches, d_model).
        """
        B, C, L = x.shape

        # Pad if necessary
        if self.padding is not None:
            x = self.padding(x)  # (B, C, L + stride)

        # Unfold into patches
        # (B, C, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Reshape to (B * C, num_patches, patch_len)
        x = x.permute(0, 1, 3, 2).reshape(B * C, self.num_patches, self.patch_len)

        # Project patches
        x = self.value_embedding(x)  # (B * C, num_patches, d_model)

        # Add positional encoding
        x = x + self.position_embedding(x)

        return self.dropout(x)


__all__ = [
    "PositionalEmbedding",
    "TokenEmbedding",
    "TemporalEmbedding",
    "TimeFeatureEmbedding",
    "DataEmbedding",
    "PatchEmbedding",
]
