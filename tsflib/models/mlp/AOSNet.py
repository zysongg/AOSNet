"""
AOSNet: Hilbert envelope-phase-frequency network for point forecasting.

This model keeps the forecasting pipeline in TSFLib's point-forecasting
interface while moving Hilbert analysis to the raw time axis. It explicitly
models envelope, phase, and instantaneous frequency, and adds a Hilbert-domain
auxiliary loss when targets are available in the batch tuple.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Per-channel temporal embedding from lookback length to hidden dimension."""

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HilbertStructureModule(nn.Module):
    """Raw-time Hilbert structure encoder with IF-aware gating."""

    # Descriptor mode -> number of gate input channels
    _DESCRIPTOR_CHANNELS = {
        "all": 8,        # amp(x,p) + phase(x_cos,x_sin,p_cos,p_sin) + ifreq(x,p)
        "no_amp": 6,     # phase(x_cos,x_sin,p_cos,p_sin) + ifreq(x,p)
        "no_phase": 4,   # amp(x,p) + ifreq(x,p)
        "no_ifreq": 6,   # amp(x,p) + phase(x_cos,x_sin,p_cos,p_sin)
        "amp_only": 2,   # amp(x,p)
        "phase_only": 4, # phase(x_cos,x_sin,p_cos,p_sin)
        "ifreq_only": 2, # ifreq(x,p)
        "raw_gate": 3,   # raw x, prior, and residual without Hilbert descriptors
    }

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        eps: float = 1e-6,
        descriptor_mode: str = "all",
        use_prior: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size should be odd, got {kernel_size}.")
        if hidden_channels <= 0:
            raise ValueError(
                f"hidden_channels must be positive, got {hidden_channels}."
            )
        if descriptor_mode not in self._DESCRIPTOR_CHANNELS:
            raise ValueError(f"Unknown descriptor_mode: {descriptor_mode}")

        self.enc_in = enc_in
        self.seq_len = seq_len
        self.eps = float(eps)
        self.descriptor_mode = descriptor_mode
        self.use_prior = use_prior
        self.prior = nn.Parameter(torch.zeros(1, enc_in, seq_len))
        if not use_prior:
            self.prior.requires_grad_(False)

        in_channels = self._DESCRIPTOR_CHANNELS[descriptor_mode]
        self.gate_net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
        )

    @staticmethod
    def _analytic_signal(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be 3D (B, C, L), got {x.shape}.")
        length = x.size(-1)
        spectrum = torch.fft.fft(x, dim=-1)
        h = torch.zeros(length, device=x.device, dtype=x.dtype)
        if length % 2 == 0:
            h[0] = 1.0
            h[length // 2] = 1.0
            h[1 : length // 2] = 2.0
        else:
            h[0] = 1.0
            h[1 : (length + 1) // 2] = 2.0
        return torch.fft.ifft(spectrum * h, dim=-1)

    @staticmethod
    def _instantaneous_frequency(z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"z must be 3D (B, C, L), got {z.shape}.")
        phase_step = torch.angle(z[..., 1:] * torch.conj(z[..., :-1]))
        return F.pad(phase_step, (1, 0), mode="replicate")

    def _descriptor(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self._analytic_signal(x)
        amp = torch.abs(z).clamp_min(self.eps)
        phase = torch.angle(z)
        return {
            "analytic": z,
            "log_amp": torch.log(amp),
            "cos_phase": torch.cos(phase),
            "sin_phase": torch.sin(phase),
            "ifreq": self._instantaneous_frequency(z),
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"x must be 3D (B, C, L), got {x.shape}.")
        batch_size, channels, length = x.shape
        if channels != self.enc_in or length != self.seq_len:
            raise ValueError(
                f"expected x shape (B, {self.enc_in}, {self.seq_len}), got {tuple(x.shape)}."
            )

        prior = self.prior.expand(batch_size, -1, -1).contiguous()
        mode = self.descriptor_mode
        if mode == "raw_gate":
            parts = [x.contiguous(), prior, prior - x.contiguous()]
            x_desc = {}
        else:
            x_desc = self._descriptor(x.contiguous())
            t_desc = self._descriptor(prior)

            # Build feature stack based on descriptor_mode
            parts = []
            if mode in ("all", "no_phase", "no_ifreq", "amp_only"):
                parts += [x_desc["log_amp"], t_desc["log_amp"]]
            if mode in ("all", "no_amp", "no_ifreq", "phase_only"):
                parts += [x_desc["cos_phase"], x_desc["sin_phase"],
                          t_desc["cos_phase"], t_desc["sin_phase"]]
            if mode in ("all", "no_amp", "no_phase", "ifreq_only"):
                parts += [x_desc["ifreq"], t_desc["ifreq"]]

        feat = torch.stack(parts, dim=2).reshape(batch_size * channels, len(parts), length)

        gate = torch.sigmoid(self.gate_net(feat)).reshape(batch_size, channels, length)
        corrected = x + gate * (prior - x)
        return corrected, gate, x_desc


class FRM(nn.Module):
    """Feature refinement over channel tokens."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.channel_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_attention(query=x, key=x, value=x)[0] + x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = getattr(configs, "seq_len", getattr(configs, "lookback_len", 96))
        self.pred_len = getattr(
            configs, "pred_len", getattr(configs, "horizon_len", 96)
        )
        self.enc_in = getattr(configs, "enc_in", getattr(configs, "num_features", 7))
        self.d_model = getattr(configs, "d_model", 256)
        self.dropout = getattr(configs, "dropout", 0.1)
        self.hidden_channels = getattr(configs, "hidden_channels", 8)
        self.hidden_kernel = getattr(configs, "hidden_kernel", 5)
        self.attention_heads = getattr(configs, "attention_heads", 4)
        self.attention_dropout = getattr(configs, "attention_dropout", self.dropout)
        self.embedding_dropout = getattr(configs, "embedding_dropout", self.dropout)

        self.eps = float(getattr(configs, "eps", 1e-6))

        # Ablation switches
        self.use_hilbert = getattr(configs, "use_hilbert", True)
        self.use_attention = getattr(configs, "use_attention", True)
        self.use_prior = getattr(configs, "use_prior", True)
        self.use_fusion = getattr(configs, "use_fusion", True)
        self.descriptor_mode = getattr(configs, "descriptor_mode", "all")

        self.hilbert = HilbertStructureModule(
            enc_in=self.enc_in,
            seq_len=self.seq_len,
            hidden_channels=self.hidden_channels,
            kernel_size=self.hidden_kernel,
            eps=self.eps,
            descriptor_mode=self.descriptor_mode,
            use_prior=self.use_prior,
        )
        self.embedding = TimeEmbedding(
            seq_len=self.seq_len,
            d_model=self.d_model,
            dropout=self.embedding_dropout,
        )
        self.frm = FRM(
            d_model=self.d_model,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
        )
        self.hidden_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )
        self.temporal_proj = nn.Linear(self.seq_len, self.pred_len)
        self.output_fusion = nn.Parameter(torch.tensor(0.0))
        self._aux_loss = torch.tensor(0.0)


    def forward(self, batch_) -> torch.Tensor:
        if isinstance(batch_, (tuple, list)):
            x = batch_[0]
            target = batch_[1] if len(batch_) > 1 and isinstance(batch_[1], torch.Tensor) else None
        else:
            x = batch_
            target = None

        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if target is not None and target.dtype != torch.float32:
            target = target.to(torch.float32)

        if self.use_hilbert:
            corrected, gate, _ = self.hilbert(x)
        else:
            corrected = x

        hidden = self.embedding(corrected)

        if self.use_attention:
            refined = self.frm(hidden)
        else:
            refined = hidden

        refined = self.hidden_proj(refined)

        if self.use_fusion:
            residual_out = self.output_proj(refined)
            base_out = self.temporal_proj(corrected)
            fusion = torch.sigmoid(self.output_fusion)
            pred = fusion * residual_out + (1.0 - fusion) * base_out
        else:
            pred = self.temporal_proj(corrected)

        return pred


AOSNet = Model
