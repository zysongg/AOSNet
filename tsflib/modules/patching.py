import torch
import torch.nn as nn
import einx


class PatchingLayer(nn.Module):
    """Time-series patching/downsampling view layer.

    - patch: unfold the last dimension into overlapping/non-overlapping patches.
    - downsample: reshape the last dimension by stride chunks (no overlap).

    Shapes:
      Input: [b, c, l]
      patch + channel_independent=True  -> [b*c, n, p]
      patch + channel_independent=False -> [b, c, n, p]
      downsample + channel_independent=True  -> [b*c, s, n]
      downsample + channel_independent=False -> [b, c, s, n]
    where l = sequence length, p = patch_len, s = stride, n = number of patches/chunks.
    """

    def __init__(
        self,
        stride: int = 12,
        patch_len: int = 12,
        patching_type: str = "patch",
        channel_independent: bool = True,
    ) -> None:
        super(PatchingLayer, self).__init__()

        if stride <= 0:
            raise ValueError("stride must be a positive integer")
        if patch_len <= 0:
            raise ValueError("patch_len must be a positive integer")

        if patching_type not in {"patch", "downsample"}:
            raise ValueError("patching_type must be 'patch' or 'downsample'")

        self.stride = stride
        self.patch_len = patch_len
        self.type = patching_type
        self.channel_independent = channel_independent
        self.num_features = None  # set at forward time

    def forward_view(self, x: torch.Tensor) -> torch.Tensor:
        """Create a view of the input by patching or downsampling.

        Args:
            x: Tensor of shape [b, c, l]

        Returns:
            Tensor with shapes described in the class docstring.
        """
        # x: [batch_size, n_vars, seq_len]
        if x.ndim != 3:
            raise ValueError("Input x must be 3D tensor of shape [b, c, l]")

        self.num_features = x.size(1)
        seq_len = x.size(-1)

        if self.type == "patch":
            if seq_len < self.patch_len:
                raise ValueError(
                    f"seq_len ({seq_len}) must be >= patch_len ({self.patch_len}) in 'patch' mode"
                )

            # [b, c, n, p]
            x_origin = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

            if self.channel_independent:
                # [b*c, n, p]
                return einx.rearrange("b c n p -> (b c) n p", x_origin)
            else:
                # [b, c, n, p]
                return x_origin
        else:  # downsample
            if seq_len % self.stride != 0:
                raise ValueError(
                    f"seq_len ({seq_len}) must be divisible by stride ({self.stride}) in 'downsample' mode"
                )

            if self.channel_independent:
                return einx.rearrange("b c (n s) -> (b c) s n", x, s=self.stride)
            else:
                return einx.rearrange("b c (n s) -> b c s n", x, s=self.stride)

    def reverse_view(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the view back to [b, c, l]-like layout by flattening.

        Note: For 'patch' with overlap (stride < patch_len), this simply
        concatenates patches and is not a perfect inverse reconstruction.
        """

        if self.num_features is None:
            raise RuntimeError(
                "num_features is unknown; call forward_view before reverse_view"
            )

        if self.type == "patch":
            if self.channel_independent:
                y = einx.rearrange(
                    "(b c) n p -> b c (n p)",
                    x,
                    c=self.num_features,
                )  # [b c l*] ,if stride==patch_len, l*=l
            else:
                y = einx.rearrange(
                    "b c n p -> b c (n p)",
                    x,
                    c=self.num_features,
                )  # [b c l*] ,if stride==patch_len, l*=l

        elif self.type == "downsample":
            # x: [b*c, n, s]
            if self.channel_independent:
                y = einx.rearrange(
                    "(b c) s n -> b c (n s)",
                    x,
                    c=self.num_features,
                )  # [b c l]
            else:
                y = einx.rearrange(
                    "b c s n -> b c (n s)",
                    x,
                    c=self.num_features,
                )  # [b c l]
        return y


if __name__ == "__main__":
    x = torch.randn(16, 3, 96)  # [B, C, T]

    # Downsample, channel_independent=True
    layer = PatchingLayer(
        stride=24, patch_len=24, patching_type="downsample", channel_independent=True
    )
    v = layer.forward_view(x)
    print("downsample, indep=True ->", v.shape)  # [b*c, s, n]
    r = layer.reverse_view(v)
    print("reverse ->", r.shape)  # [b, c, l]

    # Patch, channel_independent=True
    layer = PatchingLayer(
        stride=24, patch_len=24, patching_type="patch", channel_independent=True
    )
    v = layer.forward_view(x)
    print("patch, indep=True ->", v.shape)  # [b*c, n, p]
    r = layer.reverse_view(v)
    print("reverse ->", r.shape)  # [b, c, l*]

    # Patch, channel_independent=False
    layer = PatchingLayer(
        stride=24, patch_len=24, patching_type="patch", channel_independent=False
    )
    v = layer.forward_view(x)
    print("patch, indep=False ->", v.shape)  # [b, c, n, p]
    r = layer.reverse_view(v)
    print("reverse ->", r.shape)  # [b, c, l*]
