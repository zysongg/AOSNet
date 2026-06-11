"""
TSFLib Layers Module

This module provides custom neural network layers for time series forecasting.
"""

from tsflib.layers.attention import (
    AttentionLayer,
    AutoCorrelation,
    AutoCorrelationLayer,
    FullAttention,
    ProbAttention,
    TriangularCausalMask,
    ProbMask,
)
from tsflib.layers.decomposition import (
    moving_avg,
    series_decomp,
    EMA,
    DEMA,
    SMA,
    LMA,
    DECOMP,
    list_ma_types,
    get_ma,
)
from tsflib.layers.diffusion import (
    DiffusionEmbedding,
    ResidualBlock,
    diff_CSDI,
)
from tsflib.layers.embed import (
    DataEmbedding,
    DataEmbedding_inverted,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_pos_temp,
    DataEmbedding_wo_temp,
    FixedEmbedding,
    PatchEmbedding,
    PositionalEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    TokenEmbedding,
)
from tsflib.layers.encoder_decoder import (
    ConvLayer,
    EncoderLayer,
    Encoder,
    DecoderLayer,
    Decoder,
    my_Layernorm,
    series_decomp_multi,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from tsflib.layers.fourier import (
    FourierBlock,
    FourierCrossAttention,
    get_frequency_modes,
)
from tsflib.layers.wavelet import (
    MultiWaveletCross,
    MultiWaveletTransform,
    FourierCrossAttentionW,
    sparseKernelFT1d,
    MWT_CZ1d,
    get_filter,
    get_phi_psi,
)
from tsflib.layers.normalization import Normalize
from tsflib.layers.cycle import CycleLayer

__all__ = [
    # Attention mechanisms
    "AttentionLayer",
    "AutoCorrelation",
    "AutoCorrelationLayer",
    "FullAttention",
    "ProbAttention",
    "TriangularCausalMask",
    "ProbMask",
    # Decomposition layers
    "moving_avg",
    "series_decomp",
    "EMA",
    "DEMA",
    "SMA",
    "LMA",
    "DECOMP",
    "list_ma_types",
    "get_ma",
    # Diffusion layers
    "DiffusionEmbedding",
    "ResidualBlock",
    "diff_CSDI",
    # Embedding layers
    "DataEmbedding",
    "DataEmbedding_inverted",
    "DataEmbedding_wo_pos",
    "DataEmbedding_wo_pos_temp",
    "DataEmbedding_wo_temp",
    "FixedEmbedding",
    "PatchEmbedding",
    "PositionalEmbedding",
    "TemporalEmbedding",
    "TimeFeatureEmbedding",
    "TokenEmbedding",
    # Encoder-Decoder layers
    "ConvLayer",
    "EncoderLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    "my_Layernorm",
    "series_decomp_multi",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    # Fourier layers
    "FourierBlock",
    "FourierCrossAttention",
    "get_frequency_modes",
    # Multiwavelet layers
    "MultiWaveletCross",
    "MultiWaveletTransform",
    "FourierCrossAttentionW",
    "sparseKernelFT1d",
    "MWT_CZ1d",
    "get_filter",
    "get_phi_psi",
    # Normalization layers
    "Normalize",
    # Cycle layers
    "CycleLayer",
]
