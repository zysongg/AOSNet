"""
Phase-alignment helpers for periodic sequences.

This module provides a lightweight alignment utility that shifts a window
so that phase 0 starts at the first position.  It is intentionally separate
from ``phase_cycle.py`` and does not change any existing source files.

Dimension convention follows the rest of TSFLib:
``(B, K, T)`` where ``T`` is the time dimension.
"""

from __future__ import annotations

import torch


def _normalize_index(index, batch_size: int, device: torch.device) -> torch.Tensor:
    """Convert ``index`` to a batched ``LongTensor`` on the target device."""
    idx = torch.as_tensor(index, device=device).long().reshape(-1)

    if idx.numel() == 1:
        return idx.expand(batch_size)

    if idx.numel() != batch_size:
        raise ValueError(
            f"index has {idx.numel()} elements but batch size is {batch_size}"
        )

    return idx


def align_phase(
    data: torch.Tensor,
    index,
    cycle_len: int,
    length: int | None = None,
) -> torch.Tensor:
    """Align a window so that phase 0 is placed at the first position.

    Args:
        data: Input tensor with shape ``(B, K, T)``, ``(K, T)``, or ``(T,)``.
        index: Absolute position of the first element in the original series.
            This determines the phase of the first point in ``data``.
        cycle_len: Period length.
        length: Number of aligned points to return.  If ``None``, the full
            window length ``T`` is returned after phase alignment.

    Returns:
        A tensor with the same leading dimensions as ``data`` and a time
        dimension of ``length`` (or ``T`` when ``length`` is ``None``).

    Example:
        >>> import torch
        >>> from tsflib.utils.phase_align import align_phase
        >>> x = torch.tensor([[[1., 2., 3., 4., 5., 6.]]])  # (B=1, K=1, T=6)
        >>> align_phase(x, index=torch.tensor([3]), cycle_len=4, length=4)
        tensor([[[2., 3., 4., 5.]]])
    """
    if cycle_len <= 0:
        raise ValueError(f"cycle_len must be positive, got {cycle_len}")

    squeeze_b = False
    squeeze_k = False

    if data.dim() == 1:
        data = data.unsqueeze(0).unsqueeze(0)
        squeeze_b = True
        squeeze_k = True
    elif data.dim() == 2:
        data = data.unsqueeze(0)
        squeeze_b = True
    elif data.dim() != 3:
        raise ValueError(
            "data must have shape (T,), (K, T), or (B, K, T); "
            f"got {tuple(data.shape)}"
        )

    B, K, T = data.shape
    if T <= 0:
        raise ValueError("time dimension must be positive")

    if length is None:
        length = T
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    index = _normalize_index(index, B, data.device)

    # Position of phase 0 inside each window.
    phase0 = (cycle_len - index % cycle_len) % cycle_len  # (B,)

    offsets = torch.arange(length, device=data.device)
    gather_idx = (phase0.unsqueeze(1) + offsets.unsqueeze(0)) % T  # (B, length)
    gather_idx = gather_idx.unsqueeze(1).expand(B, K, length)

    aligned = torch.gather(data, 2, gather_idx)

    if squeeze_b and squeeze_k:
        return aligned.squeeze(0).squeeze(0)
    if squeeze_b:
        return aligned.squeeze(0)
    return aligned


__all__ = ["align_phase"]
