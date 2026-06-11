"""
TQNet: Temporal Query Network for Time Series Forecasting

This is a reimplementation adapted for TSFLib.
"""

import torch
import torch.nn as nn

from tsflib.layers import CycleLayer


class Model(nn.Module):
    """TQNet model for time series forecasting.

    TQNet uses temporal queries with cycle-based patterns.

    Args:
        configs: Configuration object with attributes:
            - num_features: Number of input features
            - lookback_len: Length of input sequence
            - horizon_len: Length of forecast horizon
            - cycle: Cycle length
            - d_model: Hidden dimension
            - dropout: Dropout rate

    Example:
        >>> configs = Configs(num_features=7, lookback_len=96, horizon_len=96)
        >>> model = Model(configs)
        >>> x = torch.randn(32, 7, 96)  # (batch, features, seq_len)
        >>> y = model(x)  # (batch, features, horizon)
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = getattr(configs, 'lookback_len', 96)
        self.pred_len = getattr(configs, 'horizon_len', 96)
        self.enc_in = getattr(configs, 'num_features', 7)
        self.cycle_len = getattr(configs, 'cycle', 24)
        self.d_model = getattr(configs, 'd_model', 256)
        self.dropout = getattr(configs, 'dropout', 0.1)

        self.use_tq = True
        self.channel_aggre = True

        if self.use_tq:
            self.temporalQuery = CycleLayer(self.cycle_len, self.enc_in)

        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(
                embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5
            )

        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout), nn.Linear(self.d_model, self.pred_len)
        )

    def forecast(self, x, cycle_index):
        if self.use_tq:
            query_input = self.temporalQuery(cycle_index, self.seq_len)
            if self.channel_aggre:
                channel_information = self.channelAggregator(
                    query=query_input, key=x, value=x
                )[0]
            else:
                channel_information = query_input
        else:
            if self.channel_aggre:
                channel_information = self.channelAggregator(
                    query=x, key=x, value=x
                )[0]
            else:
                channel_information = 0

        input_proj = self.input_proj(x + channel_information)
        hidden = self.model(input_proj)
        output = self.output_proj(hidden + input_proj)

        return output

    def forward(self, batch_) -> torch.Tensor:
        """Forward pass.

        Args:
            batch_: Tuple of (x, y, x_mark, y_mark, index) where:
                x: Input tensor of shape (B, C, L)
                index: Absolute position in the full time series (used for cycle_index)

        Returns:
            Output tensor of shape (B, C, H) where H = horizon length
        """
        x, batch_y, batch_x_mark, batch_y_mark, index = batch_

        # Ensure input is float32
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Compute cycle_index from absolute position
        cycle_index = index.long() % self.cycle_len

        output = self.forecast(x, cycle_index)
        return output


# Alias for easy import
TQNet = Model
