"""
Default Configurations for MLP-based Models

Provides dataclass-based configurations separated into logical groups:
- NormConfig: Normalization settings
- LossConfig: Loss function settings
- ModelConfig: Model-specific hyperparameters
- DataConfig: Data-related settings (lookback, horizon, etc.)

Each model has a ready-to-use config that can be imported directly.

Example:
    >>> from tsflib.confs.mlp import DLinearConfig
    >>> config = DLinearConfig(lookback=96, horizon=96, num_features=7)
    >>> print(config.model)
    >>> print(config.norm)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Base config dataclasses
# --------------------------------------------------------------------------

@dataclass
class NormConfig:
    """Normalization configuration.

    Attributes:
        use_norm: Whether to use normalization (RevIN).
        affine: Whether RevIN uses learnable affine parameters.
        subtract_last: Whether to subtract last value instead of mean.
        time_dim: Index of time dimension (default -1).
        feature_dim: Index of feature dimension (default 1).
    """
    use_norm: bool = True
    affine: bool = False
    subtract_last: bool = False
    time_dim: int = -1
    feature_dim: int = 1


@dataclass
class LossConfig:
    """Loss function configuration.

    Attributes:
        loss_type: Loss specification. Can be:
            - str: Single loss name, e.g. "mse"
            - List: [loss_name, alpha] for single loss with config
            - List[List]: [[name, weight], ...] for multi-loss
        loss_weights: Weights for multi-loss mode.
    """
    loss_type: Any = "mse"
    loss_weights: Optional[List[float]] = None


@dataclass
class DataConfig:
    """Data-related configuration.

    Attributes:
        num_features: Number of input features (channels).
        lookback: Input sequence length.
        horizon: Forecast horizon length.
        label_len: Label length for teacher forcing (default 0).
        batch_size: Batch size for training.
    """
    num_features: int = 7
    lookback: int = 96
    horizon: int = 96
    label_len: int = 0
    batch_size: int = 32


@dataclass
class TrainConfig:
    """Training configuration.

    Attributes:
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        epochs: Maximum training epochs.
        scheduler_type: LR scheduler type ('plateau', 'step', 'cosine').
        scheduler_patience: Patience for ReduceLROnPlateau.
        scheduler_factor: Factor for LR reduction.
        early_stopping_patience: Early stopping patience.
        gpu: GPU device index.
    """
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    scheduler_type: str = "plateau"
    scheduler_patience: int = 1
    scheduler_factor: float = 0.3
    early_stopping_patience: int = 3
    gpu: int = 0


# --------------------------------------------------------------------------
# DLinear
# --------------------------------------------------------------------------

@dataclass
class DLinearModelConfig:
    """DLinear model-specific configuration.

    Attributes:
        individual: Whether to use individual linear layers per feature.
    """
    individual: bool = False


@dataclass
class DLinearConfig:
    """Complete configuration for DLinear."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: DLinearModelConfig = field(default_factory=DLinearModelConfig)


# --------------------------------------------------------------------------
# NLinear
# --------------------------------------------------------------------------

@dataclass
class NLinearModelConfig:
    """NLinear model-specific configuration."""
    pass  # NLinear has no special hyperparameters beyond data dims


@dataclass
class NLinearConfig:
    """Complete configuration for NLinear."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: NLinearModelConfig = field(default_factory=NLinearModelConfig)


# --------------------------------------------------------------------------
# Linear
# --------------------------------------------------------------------------

@dataclass
class LinearModelConfig:
    """Linear model-specific configuration."""
    d_model: int = 128


@dataclass
class LinearConfig:
    """Complete configuration for Linear."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: LinearModelConfig = field(default_factory=LinearModelConfig)


# --------------------------------------------------------------------------
# MLP
# --------------------------------------------------------------------------

@dataclass
class MLPModelConfig:
    """MLP model-specific configuration."""
    d_model: int = 128


@dataclass
class MLPConfig:
    """Complete configuration for MLP."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: MLPModelConfig = field(default_factory=MLPModelConfig)


# --------------------------------------------------------------------------
# SparseTSF
# --------------------------------------------------------------------------

@dataclass
class SparseTSFModelConfig:
    """SparseTSF model-specific configuration.

    Attributes:
        period_len: Period length for sparse sampling (downsample stride).
    """
    period_len: int = 24


@dataclass
class SparseTSFConfig:
    """Complete configuration for SparseTSF."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: SparseTSFModelConfig = field(default_factory=lambda: SparseTSFModelConfig(period_len=24))


# --------------------------------------------------------------------------
# FITS
# --------------------------------------------------------------------------

@dataclass
class FITSModelConfig:
    """FITS model-specific configuration.

    Attributes:
        H_order: Fourier expansion order.
        base_T: Base period for Fourier basis.
        cut_freq: Cutoff frequency (0 = auto).
        individual: Whether to use individual layers per feature.
    """
    H_order: int = 10
    base_T: int = 24
    cut_freq: int = 0
    individual: bool = False


@dataclass
class FITSConfig:
    """Complete configuration for FITS."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: FITSModelConfig = field(default_factory=FITSModelConfig)


# --------------------------------------------------------------------------
# TiDE
# --------------------------------------------------------------------------

@dataclass
class TiDEModelConfig:
    """TiDE model-specific configuration.

    Attributes:
        d_model: Hidden dimension.
        e_layers: Encoder layers.
        d_layers: Decoder layers.
        feature_encode_dim: Feature encoding dimension.
        d_ff: Feed-forward dimension.
        dropout: Dropout rate.
    """
    d_model: int = 256
    e_layers: int = 2
    d_layers: int = 1
    feature_encode_dim: int = 64
    d_ff: int = 512
    dropout: float = 0.1


@dataclass
class TiDEConfig:
    """Complete configuration for TiDE."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: TiDEModelConfig = field(default_factory=TiDEModelConfig)


# --------------------------------------------------------------------------
# Koopa
# --------------------------------------------------------------------------

@dataclass
class KoopaModelConfig:
    """Koopa model-specific configuration.

    Attributes:
        dynamic_dim: Dynamic state dimension.
        hidden_dim: Hidden layer dimension.
        hidden_layers: Number of hidden layers.
        mask_spectrum: Spectrum mask (None = auto).
        multistep: Whether to use multistep forecasting.
        num_blocks: Number of Koopa blocks.
        seg_len: Segment length.
    """
    dynamic_dim: int = 128
    hidden_dim: int = 256
    hidden_layers: int = 3
    mask_spectrum: Optional[List[int]] = None
    multistep: bool = False
    num_blocks: int = 4
    seg_len: int = 48


@dataclass
class KoopaConfig:
    """Complete configuration for Koopa."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: KoopaModelConfig = field(default_factory=KoopaModelConfig)


# --------------------------------------------------------------------------
# FreTS
# --------------------------------------------------------------------------

@dataclass
class FreTSModelConfig:
    """FreTS model-specific configuration.

    Attributes:
        channel_independent: Whether to process channels independently.
    """
    channel_independent: bool = False


@dataclass
class FreTSConfig:
    """Complete configuration for FreTS."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: FreTSModelConfig = field(default_factory=FreTSModelConfig)


# --------------------------------------------------------------------------
# CycleNet
# --------------------------------------------------------------------------

@dataclass
class CycleNetModelConfig:
    """CycleNet model-specific configuration.

    Attributes:
        cycle: Cycle length (e.g., 24 for daily cycles).
        d_model: Model dimension.
        dropout: Dropout rate.
        model_type: Model type ('mlp' or 'linear').
    """
    cycle: int = 24
    d_model: int = 512
    dropout: float = 0.2
    model_type: str = "mlp"


@dataclass
class CycleNetConfig:
    """Complete configuration for CycleNet."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: CycleNetModelConfig = field(default_factory=CycleNetModelConfig)


# --------------------------------------------------------------------------
# CycleNetProb
# --------------------------------------------------------------------------

@dataclass
class CycleNetProbModelConfig:
    """CycleNetProb model-specific configuration.

    Attributes:
        cycle: Cycle length.
        d_model: Model dimension.
        dropout: Dropout rate.
        model_type: Model type.
        sigma_min: Minimum noise standard deviation.
        num_samples: Number of samples for probabilistic prediction.
    """
    cycle: int = 24
    d_model: int = 512
    dropout: float = 0.2
    model_type: str = "mlp"
    sigma_min: float = 1e-4
    num_samples: int = 100


@dataclass
class CycleNetProbConfig:
    """Complete configuration for CycleNetProb."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="nll"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: CycleNetProbModelConfig = field(default_factory=CycleNetProbModelConfig)


# --------------------------------------------------------------------------
# LightTS
# --------------------------------------------------------------------------

@dataclass
class LightTSModelConfig:
    """LightTS model-specific configuration.

    Attributes:
        d_model: Model dimension.
        dropout: Dropout rate.
        num_class: Number of output classes.
        output_attention: Whether to output attention weights.
    """
    d_model: int = 128
    dropout: float = 0.1
    num_class: int = 1
    output_attention: bool = False


@dataclass
class LightTSConfig:
    """Complete configuration for LightTS."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: LightTSModelConfig = field(default_factory=LightTSModelConfig)


# --------------------------------------------------------------------------
# xPatch
# --------------------------------------------------------------------------

@dataclass
class xPatchModelConfig:
    """xPatch model-specific configuration.

    Attributes:
        patch_len: Patch length.
        stride: Patch stride.
        padding_patch: Padding for patches.
        ma_type: Moving average type ('ema', 'sma', etc.).
        alpha: MA smoothing factor.
        beta: MA trend factor.
    """
    patch_len: int = 16
    stride: int = 16
    padding_patch: int = 0
    ma_type: str = "ema"
    alpha: float = 0.1
    beta: float = 0.1


@dataclass
class xPatchConfig:
    """Complete configuration for xPatch."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: xPatchModelConfig = field(default_factory=xPatchModelConfig)


# --------------------------------------------------------------------------
# PatchMLP
# --------------------------------------------------------------------------

@dataclass
class PatchMLPModelConfig:
    """PatchMLP model-specific configuration.

    Attributes:
        d_model: Model dimension.
        e_layers: Number of encoder layers.
    """
    d_model: int = 512
    e_layers: int = 2


@dataclass
class PatchMLPConfig:
    """Complete configuration for PatchMLP."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: PatchMLPModelConfig = field(default_factory=PatchMLPModelConfig)


# --------------------------------------------------------------------------
# CrossLinear
# --------------------------------------------------------------------------

@dataclass
class CrossLinearModelConfig:
    """CrossLinear model-specific configuration.

    Attributes:
        patch_len: Patch length.
        alpha: Decomposition alpha.
        beta: Decomposition beta.
        d_model: Model dimension.
        d_ff: Feed-forward dimension.
    """
    patch_len: int = 16
    alpha: float = 0.5
    beta: float = 0.5
    d_model: int = 64
    d_ff: int = 256


@dataclass
class CrossLinearConfig:
    """Complete configuration for CrossLinear."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: CrossLinearModelConfig = field(default_factory=CrossLinearModelConfig)


# --------------------------------------------------------------------------
# LDLinear
# --------------------------------------------------------------------------

@dataclass
class LDLinearModelConfig:
    """LDLinear model-specific configuration.

    Attributes:
        individual: Whether to use individual linear layers per feature.
    """
    individual: bool = True


@dataclass
class LDLinearConfig:
    """Complete configuration for LDLinear."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: LDLinearModelConfig = field(default_factory=LDLinearModelConfig)


# --------------------------------------------------------------------------
# SOFTS
# --------------------------------------------------------------------------

@dataclass
class SOFTSModelConfig:
    """SOFTS model-specific configuration.

    Attributes:
        activation: Activation function.
        d_core: Core dimension.
        d_ff: Feed-forward dimension.
        d_model: Model dimension.
        dropout: Dropout rate.
        e_layers: Number of encoder layers.
        output_attention: Whether to output attention weights.
    """
    activation: str = "gelu"
    d_core: int = 128
    d_ff: int = 512
    d_model: int = 512
    dropout: float = 0.1
    e_layers: int = 3
    output_attention: bool = False


@dataclass
class SOFTSConfig:
    """Complete configuration for SOFTS."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: SOFTSModelConfig = field(default_factory=SOFTSModelConfig)


# --------------------------------------------------------------------------
# Amplifier
# --------------------------------------------------------------------------

@dataclass
class AmplifierModelConfig:
    """Amplifier model-specific configuration.

    Attributes:
        hidden_size: Hidden layer size.
        kernel_size: Kernel size for decomposition.
        use_sci: Whether to use SCI mode.
    """
    hidden_size: int = 256
    kernel_size: int = 25
    use_sci: bool = False


@dataclass
class AmplifierConfig:
    """Complete configuration for Amplifier."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: AmplifierModelConfig = field(default_factory=AmplifierModelConfig)


# --------------------------------------------------------------------------
# FIPRNet  (Frequency-domain Interpolation & Residual Network)
# --------------------------------------------------------------------------

@dataclass
class FIPRNetModelConfig:
    """FIPRNet model-specific configuration.

    Attributes:
        trend_freq: Trend frequency ('auto' or int or float in (0,1]).
        stride: Stride for temporal token downsampling.
        d_model: Hidden dimension of the residual MLP.
    """
    trend_freq: Any = "auto"
    stride: int = 24
    d_model: int = 128


@dataclass
class FIPRNetConfig:
    """Complete configuration for FIPRNet."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: FIPRNetModelConfig = field(default_factory=FIPRNetModelConfig)


# Backward-compatible aliases
MURAModelConfig = FIPRNetModelConfig
MURAConfig = FIPRNetConfig


# --------------------------------------------------------------------------
# SRSNet
# --------------------------------------------------------------------------

@dataclass
class SRSNetModelConfig:
    """SRSNet model-specific configuration.

    Attributes:
        hidden_size: Hidden layer size.
        d_model: Model dimension.
        patch_len: Patch length.
        stride: Patch stride.
        dropout: Dropout rate.
        head_mode: Head mode ('mlp', 'linear', etc.).
        alpha: Scaling factor.
        pos: Whether to use positional encoding.
    """
    hidden_size: int = 128
    d_model: int = 512
    patch_len: int = 24
    stride: int = 24
    dropout: float = 0.1
    head_mode: str = "mlp"
    alpha: float = 3.5
    pos: bool = True


@dataclass
class SRSNetConfig:
    """Complete configuration for SRSNet."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: SRSNetModelConfig = field(default_factory=SRSNetModelConfig)


# --------------------------------------------------------------------------
# TQNet
# --------------------------------------------------------------------------

@dataclass
class TQNetModelConfig:
    """TQNet model-specific configuration.

    Attributes:
        cycle: Cycle length.
        d_model: Model dimension.
        dropout: Dropout rate.
    """
    cycle: int = 24
    d_model: int = 512
    dropout: float = 0.5


@dataclass
class TQNetConfig:
    """Complete configuration for TQNet."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: TQNetModelConfig = field(default_factory=TQNetModelConfig)


# --------------------------------------------------------------------------
# TimeEmb
# --------------------------------------------------------------------------

@dataclass
class TimeEmbModelConfig:
    """TimeEmb model-specific configuration.

    Attributes:
        d_model: Hidden dimension.
        use_hour_index: Whether to use the hourly temporal embedding.
        use_day_index: Whether to use the daily/weekday temporal embedding.
        hour_length: Number of hour/phase buckets.
        day_length: Number of day buckets.
        scale: Initialization scale for the frequency-domain weight.
    """
    d_model: int = 256
    use_hour_index: bool = True
    use_day_index: bool = False
    hour_length: int = 24
    day_length: int = 7
    scale: float = 0.02


@dataclass
class TimeEmbConfig:
    """Complete configuration for TimeEmb."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: TimeEmbModelConfig = field(default_factory=TimeEmbModelConfig)


# --------------------------------------------------------------------------
# TimeKAN
# --------------------------------------------------------------------------

@dataclass
class TimeKANModelConfig:
    """TimeKAN model-specific configuration.

    Attributes:
        d_model: Model dimension.
        dropout: Dropout rate.
        e_layers: Number of encoder layers.
        down_sampling_window: Down-sampling window size.
        down_sampling_layers: Number of down-sampling layers.
        begin_order: Beginning polynomial order.
        moving_avg: Moving average kernel size.
        embed: Embedding type.
        freq: Data frequency.
        use_norm: Whether to use normalization.
        channel_independence: Whether to use channel independence.
    """
    d_model: int = 32
    dropout: float = 0.1
    e_layers: int = 2
    down_sampling_window: int = 2
    down_sampling_layers: int = 3
    begin_order: int = 3
    moving_avg: int = 25
    embed: str = "timeF"
    freq: str = "h"
    use_norm: int = 1
    channel_independence: int = 1


@dataclass
class TimeKANConfig:
    """Complete configuration for TimeKAN."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: TimeKANModelConfig = field(default_factory=TimeKANModelConfig)


# --------------------------------------------------------------------------
# NBEATS
# --------------------------------------------------------------------------

@dataclass
class NBEATSModelConfig:
    """NBEATS model-specific configuration.

    Attributes:
        num_stacks: Number of stacks.
        num_blocks: Number of blocks per stack.
        num_layers: Number of layers per block.
        d_model: Model dimension.
        expansion_coeff: Expansion coefficient.
    """
    num_stacks: int = 2
    num_blocks: int = 1
    num_layers: int = 2
    d_model: int = 256
    expansion_coeff: int = 10


@dataclass
class NBEATSConfig:
    """Complete configuration for NBEATS."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: NBEATSModelConfig = field(default_factory=NBEATSModelConfig)


# --------------------------------------------------------------------------
# PRISM
# --------------------------------------------------------------------------

@dataclass
class PRISMModelConfig:
    """PRISM model-specific configuration.

    Attributes:
        period: Main period length P (must divide lookback_len).
        d_model: Model dimension.
        num_selected: Number of phase subsequences to select.
        dropout: Dropout rate.
        hidden_size: Hidden size for scorer and stats MLP.
        n_heads: Number of attention heads for cross-phase attention.
        lambda_global: Weight for global score in dual-level scoring.
        beta_ema: EMA momentum for global CLS update.
        gamma_contrast: Weight for contrastive loss.
        use_contrast: Whether to use contrastive loss.
    """
    period: int = 24
    d_model: int = 256
    num_selected: int = 8
    dropout: float = 0.1
    hidden_size: int = 64
    n_heads: int = 4
    lambda_global: float = 1.0
    beta_ema: float = 0.99
    gamma_contrast: float = 0.1
    use_contrast: bool = True


@dataclass
class PRISMConfig:
    """Complete configuration for PRISM."""
    norm: NormConfig = field(default_factory=lambda: NormConfig(use_norm=True, affine=False))
    loss: LossConfig = field(default_factory=lambda: LossConfig(loss_type="mse"))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: PRISMModelConfig = field(default_factory=PRISMModelConfig)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

MODEL_CONFIG_REGISTRY = {
    "DLinear": (DLinearConfig, DLinearModelConfig),
    "NLinear": (NLinearConfig, NLinearModelConfig),
    "Linear": (LinearConfig, LinearModelConfig),
    "MLP": (MLPConfig, MLPModelConfig),
    "SparseTSF": (SparseTSFConfig, SparseTSFModelConfig),
    "FITS": (FITSConfig, FITSModelConfig),
    "TiDE": (TiDEConfig, TiDEModelConfig),
    "Koopa": (KoopaConfig, KoopaModelConfig),
    "FreTS": (FreTSConfig, FreTSModelConfig),
    "CycleNet": (CycleNetConfig, CycleNetModelConfig),
    "CycleNetProb": (CycleNetProbConfig, CycleNetProbModelConfig),
    "LightTS": (LightTSConfig, LightTSModelConfig),
    "xPatch": (xPatchConfig, xPatchModelConfig),
    "PatchMLP": (PatchMLPConfig, PatchMLPModelConfig),
    "CrossLinear": (CrossLinearConfig, CrossLinearModelConfig),
    "LDLinear": (LDLinearConfig, LDLinearModelConfig),
    "SOFTS": (SOFTSConfig, SOFTSModelConfig),
    "Amplifier": (AmplifierConfig, AmplifierModelConfig),
    "FIPRNet": (FIPRNetConfig, FIPRNetModelConfig),
    "MURA": (FIPRNetConfig, FIPRNetModelConfig),  # backward-compat alias
    "SRSNet": (SRSNetConfig, SRSNetModelConfig),
    "TQNet": (TQNetConfig, TQNetModelConfig),
    "TimeEmb": (TimeEmbConfig, TimeEmbModelConfig),
    "TimeKAN": (TimeKANConfig, TimeKANModelConfig),
    "NBEATS": (NBEATSConfig, NBEATSModelConfig),
    "PRISM": (PRISMConfig, PRISMModelConfig),
}


def get_config(name: str, **kwargs) -> Any:
    """Get a complete config for an MLP model by name.

    Args:
        name: Model name (e.g., 'DLinear', 'SparseTSF').
        **kwargs: Override any data field (lookback, horizon, num_features, etc.).

    Returns:
        Complete config object with norm, loss, data, train, model attributes.

    Example:
        >>> config = get_config("DLinear", lookback=336, horizon=96, num_features=7)
        >>> print(config.data.lookback)  # 336
        >>> print(config.model.individual)  # False
    """
    if name not in MODEL_CONFIG_REGISTRY:
        available = ", ".join(sorted(MODEL_CONFIG_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")

    config_cls, _ = MODEL_CONFIG_REGISTRY[name]
    config = config_cls()

    # Apply overrides to data config if provided
    if "lookback" in kwargs:
        config.data.lookback = kwargs["lookback"]
    if "horizon" in kwargs:
        config.data.horizon = kwargs["horizon"]
    if "num_features" in kwargs:
        config.data.num_features = kwargs["num_features"]
    if "batch_size" in kwargs:
        config.data.batch_size = kwargs["batch_size"]

    # Apply overrides to train config if provided
    if "lr" in kwargs:
        config.train.lr = kwargs["lr"]
    if "epochs" in kwargs:
        config.train.epochs = kwargs["epochs"]

    return config


def list_models() -> list:
    """List all available MLP-based model names."""
    return sorted(MODEL_CONFIG_REGISTRY.keys())


__all__ = [
    # Base configs
    "NormConfig",
    "LossConfig",
    "DataConfig",
    "TrainConfig",
    # Model configs
    "DLinearConfig",
    "DLinearModelConfig",
    "NLinearConfig",
    "NLinearModelConfig",
    "LinearConfig",
    "LinearModelConfig",
    "MLPConfig",
    "MLPModelConfig",
    "SparseTSFConfig",
    "SparseTSFModelConfig",
    "FITSConfig",
    "FITSModelConfig",
    "TiDEConfig",
    "TiDEModelConfig",
    "KoopaConfig",
    "KoopaModelConfig",
    "FreTSConfig",
    "FreTSModelConfig",
    "CycleNetConfig",
    "CycleNetModelConfig",
    "CycleNetProbConfig",
    "CycleNetProbModelConfig",
    "LightTSConfig",
    "LightTSModelConfig",
    "xPatchConfig",
    "xPatchModelConfig",
    "PatchMLPConfig",
    "PatchMLPModelConfig",
    "CrossLinearConfig",
    "CrossLinearModelConfig",
    "LDLinearConfig",
    "LDLinearModelConfig",
    "SOFTSConfig",
    "SOFTSModelConfig",
    "AmplifierConfig",
    "AmplifierModelConfig",
    "FIPRNetConfig",
    "FIPRNetModelConfig",
    "MURAConfig",
    "MURAModelConfig",
    "SRSNetConfig",
    "SRSNetModelConfig",
    "TQNetConfig",
    "TQNetModelConfig",
    "TimeEmbConfig",
    "TimeEmbModelConfig",
    "TimeKANConfig",
    "TimeKANModelConfig",
    "NBEATSConfig",
    "NBEATSModelConfig",
    "PRISMConfig",
    "PRISMModelConfig",
    # Registry
    "MODEL_CONFIG_REGISTRY",
    "get_config",
    "list_models",
]
