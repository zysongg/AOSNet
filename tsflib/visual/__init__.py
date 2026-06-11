"""
TSFLib Visualization Module

Provides functions for visualizing time series forecasting results,
including point and probabilistic predictions with lookback intervals.

Usage:
    from tsflib.visual import plot_point_forecast, plot_prob_forecast
    from tsflib.visual import plot_point_multi_channel, plot_prob_multi_channel
"""

from tsflib.visual.predictions import (
    plot_point_forecast,
    plot_point_multi_channel,
    plot_prob_forecast,
    plot_prob_multi_channel,
)

__all__ = [
    "plot_point_forecast",
    "plot_point_multi_channel",
    "plot_prob_forecast",
    "plot_prob_multi_channel",
]
