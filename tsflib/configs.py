"""
TSFLib Default Configurations

This module provides default configurations for common time series forecasting
scenarios, making it easy to get started without writing custom configs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class DataConfig:
    """Configuration for data loading.

    Attributes:
        lookback_len: Length of input sequence (history).
        horizon_len: Length of forecast horizon (prediction).
        label_len: Length of label sequence (for teacher forcing).
        batch_size: Batch size for training.
        num_workers: Number of data loading workers.
        split_ratio: Train/val/test split ratios.
        scale: Whether to apply normalization.
        timeenc: Time encoding type (0=basic, 1=frequency features).
        freq: Data frequency ('h', 't', 's', 'd', etc.).
    """
    lookback_len: int = 96
    horizon_len: int = 96
    label_len: int = 48
    batch_size: int = 32
    num_workers: int = 4
    split_ratio: List[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])
    scale: bool = True
    timeenc: int = 0
    freq: str = "h"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lookback_len": self.lookback_len,
            "horizon_len": self.horizon_len,
            "label_len": self.label_len,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "split_ratio": self.split_ratio,
            "scale": self.scale,
            "timeenc": self.timeenc,
            "freq": self.freq,
        }


@dataclass
class ModelConfig:
    """Base configuration for models.

    Attributes:
        num_features: Number of input features.
        lookback_len: Length of input sequence.
        horizon_len: Length of forecast horizon.
        individual: Whether to use individual layers per feature.
    """
    num_features: int = 7
    lookback_len: int = 96
    horizon_len: int = 96
    individual: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_features": self.num_features,
            "lookback_len": self.lookback_len,
            "horizon_len": self.horizon_len,
            "individual": self.individual,
        }


@dataclass
class TrainConfig:
    """Configuration for training.

    Attributes:
        max_epochs: Maximum number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        loss_type: Loss function type ('mse', 'mae', etc.).
        use_norm: Whether to use normalization.
        early_stopping_patience: Patience for early stopping (0 to disable).
        accelerator: Accelerator type ('cpu', 'gpu', etc.).
        devices: Number of devices to use.
    """
    max_epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss_type: str = "mse"
    use_norm: bool = False
    early_stopping_patience: int = 0
    accelerator: str = "gpu"
    devices: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "loss_type": self.loss_type,
            "use_norm": self.use_norm,
            "early_stopping_patience": self.early_stopping_patience,
            "accelerator": self.accelerator,
            "devices": self.devices,
        }


class PresetConfigs:
    """Preset configurations for common scenarios.

    This class provides factory methods for creating configurations
    for common time series forecasting scenarios.

    Example:
        >>> from tsflib.configs import PresetConfigs
        >>>
        >>> # Get ETTh1 configuration
        >>> data_cfg, model_cfg, train_cfg = PresetConfigs.etth1()
        >>>
        >>> # Get configuration with custom parameters
        >>> data_cfg, model_cfg, train_cfg = PresetConfigs.etth1(
        ...     lookback_len=192,
        ...     horizon_len=96,
        ...     batch_size=64
        ... )
    """

    @staticmethod
    def etth1(
        lookback_len: int = 96,
        horizon_len: int = 96,
        batch_size: int = 32,
        max_epochs: int = 10,
        lr: float = 1e-3,
    ) -> tuple[DataConfig, ModelConfig, TrainConfig]:
        """Configuration for ETTh1 dataset.

        ETTh1 has 7 features (Oil Temperature + 6 power load features).

        Args:
            lookback_len: Input sequence length.
            horizon_len: Forecast horizon length.
            batch_size: Batch size.
            max_epochs: Maximum training epochs.
            lr: Learning rate.

        Returns:
            Tuple of (DataConfig, ModelConfig, TrainConfig).
        """
        data_cfg = DataConfig(
            lookback_len=lookback_len,
            horizon_len=horizon_len,
            label_len=48,
            batch_size=batch_size,
            num_workers=4,
            split_ratio=[0.6, 0.2, 0.2],
            scale=True,
            timeenc=0,
            freq="h",
        )

        model_cfg = ModelConfig(
            num_features=7,
            lookback_len=lookback_len,
            horizon_len=horizon_len,
            individual=False,
        )

        train_cfg = TrainConfig(
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=0.0,
            loss_type="mse",
            use_norm=False,
            early_stopping_patience=0,
            accelerator="gpu",
            devices=1,
        )

        return data_cfg, model_cfg, train_cfg

    @staticmethod
    def ettm1(
        lookback_len: int = 96,
        horizon_len: int = 96,
        batch_size: int = 32,
        max_epochs: int = 10,
        lr: float = 1e-3,
    ) -> tuple[DataConfig, ModelConfig, TrainConfig]:
        """Configuration for ETTm1 dataset.

        ETTm1 has 7 features with 15-minute frequency.

        Args:
            lookback_len: Input sequence length.
            horizon_len: Forecast horizon length.
            batch_size: Batch size.
            max_epochs: Maximum training epochs.
            lr: Learning rate.

        Returns:
            Tuple of (DataConfig, ModelConfig, TrainConfig).
        """
        data_cfg = DataConfig(
            lookback_len=lookback_len,
            horizon_len=horizon_len,
            label_len=48,
            batch_size=batch_size,
            num_workers=4,
            split_ratio=[0.6, 0.2, 0.2],
            scale=True,
            timeenc=0,
            freq="t",  # 15-minute frequency
        )

        model_cfg = ModelConfig(
            num_features=7,
            lookback_len=lookback_len,
            horizon_len=horizon_len,
            individual=False,
        )

        train_cfg = TrainConfig(
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=0.0,
            loss_type="mse",
            use_norm=False,
            early_stopping_patience=0,
            accelerator="gpu",
            devices=1,
        )

        return data_cfg, model_cfg, train_cfg

    @staticmethod
    def weather(
        lookback_len: int = 96,
        horizon_len: int = 96,
        batch_size: int = 32,
        max_epochs: int = 10,
        lr: float = 1e-3,
    ) -> tuple[DataConfig, ModelConfig, TrainConfig]:
        """Configuration for Weather dataset.

        Weather dataset has 21 features.

        Args:
            lookback_len: Input sequence length.
            horizon_len: Forecast horizon length.
            batch_size: Batch size.
            max_epochs: Maximum training epochs.
            lr: Learning rate.

        Returns:
            Tuple of (DataConfig, ModelConfig, TrainConfig).
        """
        data_cfg = DataConfig(
            lookback_len=lookback_len,
            horizon_len=horizon_len,
            label_len=48,
            batch_size=batch_size,
            num_workers=4,
            split_ratio=[0.7, 0.1, 0.2],  # Different split for weather
            scale=True,
            timeenc=0,
            freq="h",
        )

        model_cfg = ModelConfig(
            num_features=21,
            lookback_len=lookback_len,
            horizon_len=horizon_len,
            individual=False,
        )

        train_cfg = TrainConfig(
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=0.0,
            loss_type="mse",
            use_norm=False,
            early_stopping_patience=0,
            accelerator="gpu",
            devices=1,
        )

        return data_cfg, model_cfg, train_cfg

    @staticmethod
    def quick_test(
        lookback_len: int = 96,
        horizon_len: int = 96,
    ) -> tuple[DataConfig, ModelConfig, TrainConfig]:
        """Configuration for quick testing/debugging.

        Uses small batch size and few epochs for fast iteration.

        Args:
            lookback_len: Input sequence length.
            horizon_len: Forecast horizon length.

        Returns:
            Tuple of (DataConfig, ModelConfig, TrainConfig).
        """
        data_cfg = DataConfig(
            lookback_len=lookback_len,
            horizon_len=horizon_len,
            label_len=48,
            batch_size=8,
            num_workers=0,  # No multiprocessing for debugging
            split_ratio=[0.6, 0.2, 0.2],
            scale=True,
            timeenc=0,
            freq="h",
        )

        model_cfg = ModelConfig(
            num_features=7,
            lookback_len=lookback_len,
            horizon_len=horizon_len,
            individual=False,
        )

        train_cfg = TrainConfig(
            max_epochs=2,
            lr=1e-3,
            weight_decay=0.0,
            loss_type="mse",
            use_norm=False,
            early_stopping_patience=0,
            accelerator="cpu",  # Use CPU for debugging
            devices=1,
        )

        return data_cfg, model_cfg, train_cfg


# Convenience function for getting presets
def get_preset(name: str, **kwargs) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    """Get a preset configuration by name.

    Args:
        name: Preset name ('etth1', 'ettm1', 'weather', 'quick_test').
        **kwargs: Override parameters.

    Returns:
        Tuple of (DataConfig, ModelConfig, TrainConfig).

    Example:
        >>> from tsflib.configs import get_preset
        >>> data_cfg, model_cfg, train_cfg = get_preset('etth1', batch_size=64)
    """
    presets = {
        "etth1": PresetConfigs.etth1,
        "ettm1": PresetConfigs.ettm1,
        "weather": PresetConfigs.weather,
        "quick_test": PresetConfigs.quick_test,
    }

    if name not in presets:
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")

    return presets[name](**kwargs)


__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "PresetConfigs",
    "get_preset",
]
