"""Time series decomposition layers.

Provides moving average variants and a unified decomposition entry point.

All layers accept input of shape ``(B, C, L)`` where ``L`` is the time dimension.
The ``DECOMP`` class is the recommended entry point — it wraps a moving average
method and returns ``(residual, trend)``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Literal


# ---------------------------------------------------------------------------
# Moving average primitives
# ---------------------------------------------------------------------------


class moving_avg(nn.Module):
    """Simple moving average via 1D average pooling with symmetric padding.

    Args:
        kernel_size: Window size.
        stride: Pooling stride.

    Input shape: ``(B, C, L)`` where ``L`` is the time dimension.
    """

    def __init__(self, kernel_size: int = 25, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        pad = (self.kernel_size - 1) // 2
        x = torch.cat([x[..., :1].expand(-1, -1, pad), x, x[..., -1:].expand(-1, -1, pad)], dim=-1)
        x = self.avg(x)
        return x


class EMA(nn.Module):
    """Exponential Moving Average.

    Args:
        alpha: Smoothing factor in (0, 1).

    Input shape: ``(B, C, L)`` where ``L`` is the time dimension.
    """

    def __init__(self, alpha: float):
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        t = x.size(-1)
        powers = torch.arange(t - 1, -1, -1, dtype=torch.double, device=x.device)
        weights = torch.pow(1 - self.alpha, powers)
        w = weights.clone()
        w[1:] *= self.alpha
        w = w.view(1, 1, t)
        divisor = weights.view(1, 1, t)
        out = torch.cumsum(x * w, dim=-1) / divisor
        return out.to(x.dtype)


class DEMA(nn.Module):
    """Double Exponential Moving Average (Holt's linear method).

    Args:
        alpha: Level smoothing factor.
        beta: Trend smoothing factor.

    Input shape: ``(B, C, L)`` where ``L`` is the time dimension.
    """

    def __init__(self, alpha: float, beta: float):
        super().__init__()
        if not 0 < alpha < 1 or not 0 < beta < 1:
            raise ValueError("alpha and beta must be in (0, 1)")
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, t = x.shape
        s_prev = x[:, :, 0]  # (B, C)
        b_prev = x[:, :, 1] - s_prev
        results = [s_prev.unsqueeze(-1)]
        for i in range(1, t):
            xt = x[:, :, i]  # (B, C)
            s = self.alpha * xt + (1 - self.alpha) * (s_prev + b_prev)
            b_prev = self.beta * (s - s_prev) + (1 - self.beta) * b_prev
            s_prev = s
            results.append(s.unsqueeze(-1))
        out = torch.cat(results, dim=-1)
        return out


class SMA(nn.Module):
    """Alias for :class:`moving_avg` (Simple Moving Average).

    Input shape: ``(B, C, L)`` where ``L`` is the time dimension.
    """

    def __init__(self, kernel_size: int = 25, stride: int = 1):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moving_avg(x)


class LMA(nn.Module):
    """Learnable Moving Average via depth-wise 1D convolution.

    Args:
        kernel_size: Convolution kernel size.
        stride: Convolution stride.

    Input shape: ``(B, C, L)`` where ``L`` is the time dimension.
    """

    def __init__(self, kernel_size: int = 25, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            padding_mode="replicate",
            bias=True,
        )
        self._init_gaussian_weights()

    def _init_gaussian_weights(self):
        half = self.kernel_size // 2
        indices = torch.arange(self.kernel_size, dtype=torch.float32)
        w = torch.exp(-((indices - half) ** 2) / 2)
        self.conv.weight.data = F.softmax(w.view(1, 1, -1), dim=-1)
        self.conv.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, l = x.shape
        out = self.conv(rearrange(x, "b c l -> (b c) 1 l"))
        out = rearrange(out, "(b c) 1 l -> b c l", b=b, c=c)
        return out


# ---------------------------------------------------------------------------
# Unified decomposition entry
# ---------------------------------------------------------------------------

_MA_REGISTRY = {
    "ema": EMA,
    "dema": DEMA,
    "sma": SMA,
    "lma": LMA,
}


def list_ma_types():
    """Return available moving average type names."""
    return list(_MA_REGISTRY.keys())


def get_ma(name: str, **kwargs) -> nn.Module:
    """Get a moving average layer by name.

    Args:
        name: One of ``'ema'``, ``'dema'``, ``'sma'``, ``'lma'``.
        **kwargs: Passed to the layer constructor.
    """
    if name not in _MA_REGISTRY:
        raise ValueError(f"Unknown ma_type '{name}'. Available: {list_ma_types()}")
    return _MA_REGISTRY[name](**kwargs)


class DECOMP(nn.Module):
    """Decompose a time series into residual and trend components.

    ``x = trend + residual``

    Args:
        ma_type: Moving average method — ``'ema'``, ``'dema'``, ``'sma'``, ``'lma'``.
        kernel: Window/kernel size for SMA and LMA.
        stride: Stride for SMA and LMA.
        alpha: Smoothing factor for EMA / DEMA.
        beta: Trend smoothing factor for DEMA.

    Input shape: ``(B, C, L)`` where ``L`` is the time dimension.

    Example:
        >>> decomp = DECOMP("ema", alpha=0.3)
        >>> x = torch.randn(32, 7, 96)  # (B, C, L)
        >>> residual, trend = decomp(x)
    """

    def __init__(
        self,
        ma_type: Literal["ema", "dema", "sma", "lma"] = "ema",
        time_last: bool = True,
        kernel: int = 25,
        stride: int = 1,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        super().__init__()
        self.time_last = time_last
        if ma_type in ("ema",):
            self.ma = EMA(alpha)
        elif ma_type in ("dema",):
            self.ma = DEMA(alpha, beta)
        elif ma_type in ("sma",):
            self.ma = SMA(kernel, stride)
        elif ma_type in ("lma",):
            self.ma = LMA(kernel, stride)
        else:
            raise ValueError(f"Unknown ma_type '{ma_type}'. Available: {list_ma_types()}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose input into (residual, trend)."""
        if not self.time_last:
            x = rearrange(x, "b l c -> b c l")
        trend = self.ma(x)
        res = x - trend
        if not self.time_last:
            trend = rearrange(trend, "b c l -> b l c")
            res = rearrange(res, "b c l -> b l c")
        return res, trend


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

series_decomp = DECOMP


__all__ = [
    "moving_avg",
    "EMA",
    "DEMA",
    "SMA",
    "LMA",
    "DECOMP",
    "series_decomp",
    "list_ma_types",
    "get_ma",
]
