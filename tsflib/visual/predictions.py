"""
Time Series Forecasting Visualization Functions

Plot point and probabilistic forecasting results with optional lookback intervals.
Style reference: src/utils/visual/prob_plot.py (t-distribution confidence intervals).
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Global font settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12


def _ensure_dir(path: str):
    """Ensure the directory for a file path exists."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_point_forecast(
    predictions: np.ndarray,
    targets: np.ndarray,
    inputs: Optional[np.ndarray] = None,
    sample_id: int = 0,
    channel: int = 0,
    lookback_len: int = 48,
    save_path: str = "point_forecast.png",
):
    """Plot single-channel point forecast visualization with lookback.

    Args:
        predictions: shape (N, C, H)
        targets: shape (N, C, H)
        inputs: shape (N, C, L), lookback input, optional
        sample_id: sample index
        channel: channel index
        lookback_len: number of lookback timesteps to display from the end of inputs
        save_path: output file path
    """
    pred = predictions[sample_id, channel, :]  # (H,)
    true = targets[sample_id, channel, :]  # (H,)
    H = len(pred)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Lookback segment
    has_lookback = inputs is not None and lookback_len > 0
    if has_lookback:
        inp = inputs[sample_id, channel, :]  # (L,)
        L_total = len(inp)
        disp_len = min(lookback_len, L_total)
        lookback_data = inp[-disp_len:]

        lb_time = np.arange(-disp_len, 0)
        ax.plot(lb_time, lookback_data, color="#1F77B4", linewidth=1.5, label="Lookback")
        # Connecting line
        ax.plot([-1, 0], [lookback_data[-1], true[0]], color="#1F77B4",
                linewidth=1.5, linestyle=":")
        # Divider
        ax.axvline(x=-0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    # Forecast segment
    fc_time = np.arange(H)
    ax.plot(fc_time, true, color="#1F77B4", linewidth=1.5,
            label="Ground Truth" if not has_lookback else "Ground Truth (forecast)")
    ax.plot(fc_time, pred, color="#D62728", linewidth=1.5, linestyle="--", label="Prediction")
    ax.fill_between(fc_time, np.minimum(pred, true), np.maximum(pred, true),
                    alpha=0.15, color="#D62728", label="Error region")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title(f"Point Forecast - Sample {sample_id}, Channel {channel}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Point] Saved: {save_path}")


def plot_point_multi_channel(
    predictions: np.ndarray,
    targets: np.ndarray,
    inputs: Optional[np.ndarray] = None,
    sample_id: int = 0,
    channels: Optional[List[int]] = None,
    lookback_len: int = 48,
    save_path: str = "point_multi_channel.png",
):
    """Plot multi-channel point forecast comparison with lookback.

    Args:
        predictions: shape (N, C, H)
        targets: shape (N, C, H)
        inputs: shape (N, C, L), optional
        sample_id: sample index
        channels: list of channel indices to plot (default: first 4)
        lookback_len: lookback display length
        save_path: output file path
    """
    C = predictions.shape[1]
    if channels is None:
        channels = list(range(min(4, C)))

    n_ch = len(channels)
    H = predictions.shape[2]
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    has_lookback = inputs is not None and lookback_len > 0

    for i, ch in enumerate(channels):
        pred = predictions[sample_id, ch, :]
        true = targets[sample_id, ch, :]

        if has_lookback:
            inp = inputs[sample_id, ch, :]
            disp_len = min(lookback_len, len(inp))
            lookback_data = inp[-disp_len:]
            lb_time = np.arange(-disp_len, 0)
            axes[i].plot(lb_time, lookback_data, color="#1F77B4", linewidth=1.2)
            axes[i].plot([-1, 0], [lookback_data[-1], true[0]], color="#1F77B4",
                         linewidth=1.2, linestyle=":")
            axes[i].axvline(x=-0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

        fc_time = np.arange(H)
        axes[i].plot(fc_time, true, color="#1F77B4", linewidth=1.2, label="Ground Truth")
        axes[i].plot(fc_time, pred, color="#D62728", linewidth=1.2, linestyle="--", label="Prediction")
        axes[i].set_ylabel(f"Channel {ch}")
        axes[i].grid(True, alpha=0.2)
        axes[i].legend(fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Time Step")
    axes[0].set_title(f"Point Forecast - Sample {sample_id}, Multi-Channel")
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Point] Multi-channel saved: {save_path}")


def plot_prob_forecast(
    samples: np.ndarray,
    targets: np.ndarray,
    inputs: Optional[np.ndarray] = None,
    sample_id: int = 0,
    channel: int = 0,
    lookback_len: int = 48,
    cl: Tuple[float, ...] = (0.5, 0.9),
    save_path: str = "prob_forecast.png",
):
    """Plot single-channel probabilistic forecast with confidence intervals and lookback.

    Uses t-distribution to compute confidence intervals from forecast samples.

    Args:
        samples: shape (N, num_samples, H, C)
        targets: shape (N, H, C)
        inputs: shape (N, L, C), optional
        sample_id: sample index
        channel: channel index
        lookback_len: lookback display length
        cl: confidence levels (e.g., (0.5, 0.9))
        save_path: output file path
    """
    y_hats = samples[sample_id, :, :, channel]  # (num_samples, H)
    ys = targets[sample_id, :, channel]  # (H,)
    H = len(ys)
    N = y_hats.shape[0]

    y_hat_median = np.median(y_hats, axis=0)
    df = N - 1
    se = stats.sem(y_hats, axis=0)
    intervals = [stats.t.interval(c, df, loc=y_hat_median, scale=se) for c in cl]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Lookback segment
    has_lookback = inputs is not None and lookback_len > 0
    if has_lookback:
        inp = inputs[sample_id, :, channel]  # (L,)
        disp_len = min(lookback_len, len(inp))
        lookback_data = inp[-disp_len:]
        lb_time = np.arange(-disp_len, 0)
        ax.plot(lb_time, lookback_data, color="#1F77B4", linewidth=1.5, label="Lookback")
        ax.plot([-1, 0], [lookback_data[-1], ys[0]], color="#1F77B4",
                linewidth=1.5, linestyle=":")
        ax.axvline(x=-0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    # Confidence intervals
    fc_time = np.arange(H)
    colors = ["#61C561", "#339933"]
    alphas = [0.6, 0.85]
    for i in reversed(range(len(cl))):
        ax.fill_between(
            fc_time, intervals[i][0], intervals[i][1],
            alpha=alphas[i], color=colors[i],
            label=f"{cl[i]*100:.0f}% CI", edgecolor="none",
        )

    ax.plot(fc_time, y_hat_median, color="#118811", linewidth=1.5, label="Median prediction")
    ax.plot(fc_time, ys, color="#1F77B4", linewidth=1.5,
            label="Ground Truth" if not has_lookback else "Ground Truth (forecast)")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title(f"Probabilistic Forecast - Sample {sample_id}, Channel {channel}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Prob] Saved: {save_path}")


def plot_prob_multi_channel(
    samples: np.ndarray,
    targets: np.ndarray,
    inputs: Optional[np.ndarray] = None,
    sample_id: int = 0,
    channels: Optional[List[int]] = None,
    lookback_len: int = 48,
    cl: Tuple[float, ...] = (0.5, 0.9),
    save_path: str = "prob_multi_channel.png",
):
    """Plot multi-channel probabilistic forecast with confidence intervals and lookback.

    Args:
        samples: shape (N, num_samples, H, C)
        targets: shape (N, H, C)
        inputs: shape (N, L, C), optional
        sample_id: sample index
        channels: list of channel indices (default: first 4)
        lookback_len: lookback display length
        cl: confidence levels
        save_path: output file path
    """
    C = targets.shape[2]
    if channels is None:
        channels = list(range(min(4, C)))

    n_ch = len(channels)
    H = targets.shape[1]
    N = samples.shape[1]

    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    colors = ["#61C561", "#339933"]
    alphas_fill = [0.6, 0.85]
    has_lookback = inputs is not None and lookback_len > 0

    for idx, ch in enumerate(channels):
        y_hats = samples[sample_id, :, :, ch]
        ys = targets[sample_id, :, ch]

        y_hat_median = np.median(y_hats, axis=0)
        df = N - 1
        se = stats.sem(y_hats, axis=0)
        intervals = [stats.t.interval(c, df, loc=y_hat_median, scale=se) for c in cl]

        if has_lookback:
            inp = inputs[sample_id, :, ch]
            disp_len = min(lookback_len, len(inp))
            lookback_data = inp[-disp_len:]
            lb_time = np.arange(-disp_len, 0)
            axes[idx].plot(lb_time, lookback_data, color="#1F77B4", linewidth=1.2)
            axes[idx].plot([-1, 0], [lookback_data[-1], ys[0]], color="#1F77B4",
                           linewidth=1.2, linestyle=":")
            axes[idx].axvline(x=-0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

        fc_time = np.arange(H)
        for i in reversed(range(len(cl))):
            axes[idx].fill_between(
                fc_time, intervals[i][0], intervals[i][1],
                alpha=alphas_fill[i], color=colors[i],
                label=f"{cl[i]*100:.0f}% CI" if idx == 0 else None,
                edgecolor="none",
            )

        axes[idx].plot(fc_time, y_hat_median, color="#118811", linewidth=1.2,
                       label="Median" if idx == 0 else None)
        axes[idx].plot(fc_time, ys, color="#1F77B4", linewidth=1.2,
                       label="Ground Truth" if idx == 0 else None)
        axes[idx].set_ylabel(f"Channel {ch}")
        axes[idx].grid(True, alpha=0.2)

    axes[0].legend(fontsize=9, loc="upper right")
    axes[-1].set_xlabel("Time Step")
    axes[0].set_title(f"Probabilistic Forecast - Sample {sample_id}, Multi-Channel")
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Prob] Multi-channel saved: {save_path}")
