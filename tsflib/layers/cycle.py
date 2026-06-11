"""
Cycle Layer for retrieving periodic patterns.

Stores a learnable full-cycle lookup table of shape (cycle_len, num_features)
and retrieves consecutive entries via modular arithmetic given a starting
cycle index.
"""

import torch
import torch.nn as nn
from tsflib.utils.phase_cycle import extract_cycle, reconstruct_cycle
import einx


class CycleLayer(nn.Module):
    """Cycle layer for retrieving periodic patterns.

    A single trainable parameter ``cycleQueue`` of shape ``(cycle_len, D)``
    represents one complete cycle.  Given the integer starting position
    *cycle_index* (where the lookback window begins inside the cycle) we
    retrieve consecutive entries via modular arithmetic.

    Args:
        cycle_len: Number of time steps in one full cycle.
        num_features: Number of channels D.
        pre_train_path: Optional path to a ``.pt``/``.pth`` file containing a
            pre-trained ``cycleQueue`` tensor.  If given, the queue is
            initialised from ``state_dict["cycleLayer.cycleQueue"]``.
        fine_tune: If True and *pre_train_path* is given, the queue remains
            trainable; otherwise it is frozen when a pre-trained path is
            provided.

    Example:
        >>> layer = CycleLayer(cycle_len=24, num_features=7)
        >>> cycle_index = torch.tensor([0, 5, 12])  # (B,)
        >>> out = layer(cycle_index, input_len=96)   # (B, D, 96)
    """

    def __init__(
        self,
        cycle_len: int,
        num_features: int,
        pre_train_path: str = None,
        fine_tune: bool = True,
    ):
        super().__init__()
        self.cycle_len = cycle_len
        self.num_features = num_features

        if pre_train_path is None:
            self.cycleQueue = nn.Parameter(
                torch.zeros((cycle_len, num_features)), requires_grad=True
            )
        else:
            data_ = torch.load(pre_train_path)["cycleLayer.cycleQueue"]
            self.cycleQueue = nn.Parameter(data_, requires_grad=fine_tune)

    def forward(self, cycle_index: torch.Tensor, input_len: int) -> torch.Tensor:
        """Retrieve cycle values starting at *cycle_index*.

        Args:
            cycle_index: Integer tensor of shape ``(B,)`` indicating where
                the lookback window starts inside the cycle.
            input_len: Number of consecutive time steps to retrieve.

        Returns:
            Tensor of shape ``(B, D, input_len)`` with the retrieved
            cycle values.
        """
        cq = einx.id("c k -> b k c", self.cycleQueue, b=cycle_index.shape[0])  # (1, cycle_len, D)
        cycle_values = reconstruct_cycle(
            cq,
            cycle_index,
            window_len=input_len,
        )  # (B, D, input_len)

        return cycle_values
