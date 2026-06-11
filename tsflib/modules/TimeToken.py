"""Time-series tokenization layer.

Splits the temporal dimension (last dim) into discrete tokens via:

- ``patch``: sliding-window extraction with optional overlap (stride < patch_len)
  or non-overlap (stride == patch_len).
- ``downsample``: non-overlapping chunking by stride.

Input shape is always ``[b, c, l]`` (batch, channels, sequence length).
"""

import torch
import torch.nn as nn
import einx
from typing import Literal


class TimeToken(nn.Module):
    """Tokenize a time-series tensor into patch/downsample tokens.

    Supported modes:

    ========================  =========================  ==========================
    Mode                      channel_independent=True   channel_independent=False
    ========================  =========================  ==========================
    patch (non-overlap)       [b*c, n, p]                [b, c, n, p]
    patch (overlap)           [b*c, n, p]                [b, c, n, p]
    downsample                [b*c, s, n]                [b, c, s, n]
    ========================  =========================  ==========================

    where ``n`` = number of tokens (``seq_len / stride``), ``p`` = patch_len,
    ``s`` = stride.  In downsample mode the last dimension is always the token
    count ``n`` so that linear layers can operate on the temporal token axis.

    Args:
        stride: Step size between consecutive patches or chunks.
        patch_len: Length of each patch (only used in ``patch`` mode).
        mode: ``"patch"`` for sliding-window, ``"downsample"`` for chunking.
        channel_independent: If True, merge batch and channel dims into ``[b*c, ...]``.

    Example:
        >>> tok = TimeToken(stride=24, patch_len=24, mode="patch", channel_independent=True)
        >>> x = torch.randn(16, 3, 96)  # [B, C, T]
        >>> tokens = tok.forward(x)      # [48, 4, 24]  (patch)
        >>> recovered = tok.reverse(tokens)  # [16, 3, 96]
    """

    def __init__(
        self,
        stride: int,
        patch_len: int = 0,
        mode: Literal["patch", "downsample"] = "patch",
        channel_independent: bool = True,
    ) -> None:
        super().__init__()

        if stride <= 0:
            raise ValueError("stride must be a positive integer")

        if mode not in {"patch", "downsample"}:
            raise ValueError("mode must be 'patch' or 'downsample'")

        if mode == "patch":
            if patch_len <= 0:
                raise ValueError("patch_len must be a positive integer in 'patch' mode")
            self.patch_len = patch_len
        else:
            # patch_len is irrelevant for downsample; store stride for symmetry
            self.patch_len = stride

        self.stride = stride
        self.mode = mode
        self.channel_independent = channel_independent
        self._num_features: int | None = None  # set during forward()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize the input tensor.

        Args:
            x: Tensor of shape ``[b, c, l]``.

        Returns:
            Tokenized tensor (shape depends on mode and channel_independent).
        """
        if x.ndim != 3:
            raise ValueError(f"Input must be 3D [b, c, l], got {x.ndim}D")

        self._num_features = x.size(1)
        seq_len = x.size(-1)

        if self.mode == "patch":
            if seq_len < self.patch_len:
                raise ValueError(
                    f"seq_len ({seq_len}) must be >= patch_len ({self.patch_len})"
                )
            tokens = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:  # downsample
            if seq_len % self.stride != 0:
                raise ValueError(
                    f"seq_len ({seq_len}) must be divisible by stride ({self.stride})"
                )
            tokens = einx.rearrange("b c (n s) -> b c s n", x, s=self.stride)

        if self.channel_independent:
            return einx.rearrange("b c ... -> (b c) ...", tokens)
        return tokens  # type: ignore[return-value]

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse tokenized tensor back to ``[b, c, l]`` layout by flattening.

        Note: In ``patch`` mode with overlap (stride < patch_len), this is **not**
        a perfect reconstruction -- patches are simply concatenated along the
        temporal dimension.

        Args:
            x: Tokenized tensor from :meth:`forward`.

        Returns:
            Tensor of shape ``[b, c, l']`` where ``l'`` may differ from the
            original length in overlapping patch mode.

        Raises:
            RuntimeError: If :meth:`forward` has not been called yet.
        """
        if self._num_features is None:
            raise RuntimeError(
                "_num_features is unknown; call forward() before reverse()"
            )

        if self.mode == "downsample":
            # Downsample stores [s, n] — flatten as n*s to recover original order
            if self.channel_independent:
                flat = einx.rearrange("(b c) s n -> b c (n s)", x, c=self._num_features, s=self.stride)
            else:
                flat = einx.rearrange("b c s n -> b c (n s)", x, s=self.stride)
        else:
            if self.channel_independent:
                flat = einx.rearrange("(b c) ... -> b c (...)", x, c=self._num_features)
            else:
                flat = einx.rearrange("b c ... -> b c (...)", x)

        return flat

    # ------------------------------------------------------------------
    # Aliases for backward compatibility
    # ------------------------------------------------------------------

    def forward_view(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(x)

    def reverse_view(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`reverse`."""
        return self.reverse(x)
