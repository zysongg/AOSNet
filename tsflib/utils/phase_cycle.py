"""
Phase-cycle utilities for periodic time series analysis.

Provides functions to extract cycle patterns from sliding windows and
reconstruct cycle values from a cycle pattern, with phase alignment
guaranteed across different window positions.

Dimension convention: all tensors use ``(B, K, T)`` layout where the
time dimension is last.

Example:
    >>> import torch
    >>> from tsflib.utils import extract_cycle, reconstruct_cycle
    >>>
    >>> # data: (B, K, T), index: (B,) absolute position of first element
    >>> cycle_pattern = extract_cycle(data, index, cycle_len=24)
    >>> cycle_values = reconstruct_cycle(cycle_pattern, index, window_len=192)
"""

from __future__ import annotations

import torch


def extract_cycle(
    data: torch.Tensor,
    index: torch.Tensor,
    cycle_len: int,
    average_cycles: bool = True,
) -> torch.Tensor:
    """Extract phase-aligned cycle pattern from windowed data.

    For each sample, finds the cycle origin (phase 0) within the window
    and extracts cycle patterns starting from there.  When the window
    contains multiple cycles (including a trailing incomplete one),
    ``average_cycles`` controls whether to average them or return the
    first one only.

    When averaging, each phase position is independently averaged over
    all available data points at that phase.  Incomplete trailing cycles
    contribute to the phases they cover, so no data is wasted.

    Phase alignment: windows with different starting indices but from the
    same periodic sequence will produce the same cycle pattern, because
    extraction always begins from phase 0.

    Args:
        data: Windowed time series, shape ``(B, K, T)`` where T is the
              total window length (e.g., lookback + horizon).
        index: Absolute position of the first element of each window in
               the original sequence, shape ``(B,)``.  Used to determine
               the phase of each element.
        cycle_len: Length of one cycle (period).
        average_cycles: If True, average all available data per phase
                        (including incomplete trailing cycles).  If False,
                        only the first cycle is used.

    Returns:
        Cycle pattern tensor of shape ``(B, K, cycle_len)``.
    """
    B, K, T = data.shape
    data = data.float()

    # Phase 0 position within the window
    cycle_start = (cycle_len - index.long() % cycle_len) % cycle_len  # (B,)

    if not average_cycles:
        # Only extract the first cycle starting from phase 0
        offsets = torch.arange(cycle_len, device=data.device)
        gather_idx = (cycle_start.unsqueeze(1) + offsets.unsqueeze(0)) % T  # (B, cycle_len)
        gather_idx = gather_idx.unsqueeze(1).expand(B, K, cycle_len)
        return torch.gather(data, 2, gather_idx)  # (B, K, cycle_len)

    # --- Per-phase averaging via mask + matmul ---
    # For each phase q, sum data[:, :, t] where phase[t] == q, then divide by count.
    positions = torch.arange(T, device=data.device)  # (T,)
    phase = (index.long().unsqueeze(1) + positions.unsqueeze(0)) % cycle_len  # (B, T)
    mask = phase.unsqueeze(2) == torch.arange(cycle_len, device=data.device).view(1, 1, cycle_len)  # (B, T, cycle_len)

    # Batch matmul: (B, K, T) @ (B, T, cycle_len) -> (B, K, cycle_len)
    result = torch.matmul(data, mask.float())
    counts = mask.sum(dim=1, keepdim=True).float()  # (B, 1, cycle_len)
    return result / counts.clamp(min=1)


def reconstruct_cycle(
    cycle_pattern: torch.Tensor,
    index: torch.Tensor,
    window_len: int,
) -> torch.Tensor:
    """Reconstruct cycle values for a window from a cycle pattern.

    This is the inverse operation of ``extract_cycle``: given a cycle
    pattern (e.g., sampled from a learned distribution), reconstruct
    the per-timestep cycle values for the entire window by repeating
    the cycle with proper phase alignment.

    Args:
        cycle_pattern: Cycle pattern tensor, shape ``(B, K, cycle_len)``.
        index: Absolute position of the first element of each window in
               the original sequence, shape ``(B,)``.
        window_len: Total window length to reconstruct (e.g., lookback
                    + horizon).

    Returns:
        Reconstructed cycle values, shape ``(B, K, window_len)``.
    """
    B, K, cycle_len = cycle_pattern.shape

    # Build per-timestep phase indices
    lens = torch.arange(window_len, device=cycle_pattern.device)
    gather_idx = (index.long().view(B, 1) + lens.view(1, -1)) % cycle_len  # (B, window_len)
    gather_idx = gather_idx.unsqueeze(1).expand(B, K, window_len)  # (B, K, window_len)

    return torch.gather(cycle_pattern, 2, gather_idx)  # (B, K, window_len)
