"""
Ready-to-use presets for MLP-based models.

Zero-config usage: just specify model name and dataset name.
All defaults can be overridden at multiple levels.

Override hierarchy (lowest to highest priority):
    1. Dataset defaults (lookback, batch_size, freq)
    2. Model presets (lr, epochs, model hyperparams)
    3. Direct params (horizon, lr, epochs, batch_size on Preset)
    4. overrides dict (any config section: model/norm/loss/train/data/wrapper/trainer)
    5. model_overrides (merged into model section, highest priority for model params)

Example:
    >>> from tsflib import train
    >>> train("AOSNet", "etth1")  # zero config

    >>> from tsflib import Preset
    >>> preset = Preset("AOSNet", "etth1", horizon=192, lr=5e-4)
    >>> preset.run()

    >>> preset = Preset("AOSNet", "etth1", overrides={
    ...     "model": {"individual": True},
    ...     "norm": {"affine": True},
    ...     "loss": {"loss_type": "mix_freq_mse"},
    ... })
    >>> preset.run()
"""

import glob
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List, Optional

import numpy as np
import torch

from tsflib.data import DataModule, load_dataset
from tsflib.data.utils import DATASET_INFO, get_data_path
from tsflib.trainers import TSTrainer, TSTrainerConfig
from tsflib.visual import (
    plot_point_forecast,
    plot_point_multi_channel,
    plot_prob_forecast,
    plot_prob_multi_channel,
)
from tsflib.wrappers import (
    TSForecastingConfig,
    TSForecastingModule,
    TSProbabilisticModule,
)

# --------------------------------------------------------------------------
# Dataset presets: auto-resolve lookback, horizon, batch_size, freq
# --------------------------------------------------------------------------

# Standard horizon presets per dataset (common in literature)
_DATASET_DEFAULTS = {
    "etth1": {
        "lookback": 96,
        "horizons": [96, 192, 336, 720],
        "batch_size": 32,
        "freq": "h",
    },
    "etth2": {
        "lookback": 96,
        "horizons": [96, 192, 336, 720],
        "batch_size": 32,
        "freq": "h",
    },
    "ettm1": {
        "lookback": 96,
        "horizons": [96, 192, 336, 720],
        "batch_size": 32,
        "freq": "t",
    },
    "ettm2": {
        "lookback": 96,
        "horizons": [96, 192, 336, 720],
        "batch_size": 32,
        "freq": "t",
    },
    "weather": {
        "lookback": 96,
        "horizons": [96, 192, 336, 720],
        "batch_size": 32,
        "freq": "t",
    },
    "traffic": {
        "lookback": 96,
        "horizons": [96, 192, 336, 720],
        "batch_size": 16,
        "freq": "h",
    },
    "electricity": {
        "lookback": 96,
        "horizons": [96, 192, 336, 720],
        "batch_size": 16,
        "freq": "h",
    },
    "exchange": {
        "lookback": 96,
        "horizons": [96, 192, 336, 720],
        "batch_size": 32,
        "freq": "d",
    },
    "solar": {
        "lookback": 96,
        "horizons": [96, 192, 336, 720],
        "batch_size": 16,
        "freq": "t",
    },
    "illness": {
        "lookback": 36,
        "horizons": [24, 36, 48, 60],
        "batch_size": 32,
        "freq": "w",
    },
}


def _get_dataset_info(dataset_name: str) -> dict:
    """Get dataset info, falling back to defaults if not in DATASET_INFO."""
    name_upper = dataset_name.upper()
    if name_upper in DATASET_INFO:
        return DATASET_INFO[name_upper]
    # Try lowercase mapping
    for key, info in DATASET_INFO.items():
        if key.lower() == dataset_name.lower():
            return info
    return {"num_features": 7, "frequency": "1h"}


def _get_dataset_defaults(dataset_name: str, horizon: int = 96) -> dict:
    """Get dataset defaults (lookback, batch_size, freq)."""
    name_lower = dataset_name.lower()
    defaults = _DATASET_DEFAULTS.get(
        name_lower, {"lookback": 96, "batch_size": 32, "freq": "h"}
    )
    return {
        "lookback": defaults["lookback"],
        "horizon": horizon,
        "batch_size": defaults["batch_size"],
        "freq": defaults["freq"],
        "sample_step": 1,  # Default sample step for probabilistic models
    }


# --------------------------------------------------------------------------
# Model presets: auto-resolve model hyperparameters
# --------------------------------------------------------------------------


_MODEL_PRESETS = {
    "CycleNet": {
        "model": {"cycle": 168, "d_model": 512, "dropout": 0.2, "model_type": "mlp"},
        "norm": {"use_norm": True, "affine": False},
        "loss": {"loss_type": "mix_freq_mse", "alpha": 1.0},
        "train": {"lr": 1e-3, "epochs": 10, "early_stopping_patience": 3},
    },
    "TQNet": {
        "model": {"cycle": 168, "d_model": 512, "dropout": 0.1},
        "norm": {"use_norm": True, "affine": False},
        "loss": {"loss_type": "mix_freq_mse", "alpha": 1.0},
        "train": {"lr": 1e-3, "epochs": 10, "early_stopping_patience": 3},
    },
    "AOSNet": {
        "model": {
            "d_model": 512,
            "dropout": 0.1,
            "hidden_channels": 8,
            "hidden_kernel": 5,
            "attention_heads": 4,
            "attention_dropout": 0.1,
            "embedding_dropout": 0.1,
            "aux_weight": 0,
            "env_weight": 1.0,
            "phase_weight": 1.0,
            "if_weight": 1.0,
        },
        "norm": {"use_norm": True, "affine": False},
        "loss": {"loss_type": "mix_freq_mse", "alpha": 1.0},
        "train": {"lr": 1e-3, "epochs": 10, "early_stopping_patience": 3},
    },
}


def _get_model_preset(model_name: str) -> dict:
    """Get model preset, raising clear error if unknown."""
    if model_name not in _MODEL_PRESETS:
        available = ", ".join(sorted(_MODEL_PRESETS.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return _MODEL_PRESETS[model_name]


# --------------------------------------------------------------------------
# Preset: the main entry point
# --------------------------------------------------------------------------


@dataclass
class Preset:
    """Ready-to-use preset for a model + dataset.

    Creates datamodule, model, and lightning module automatically.
    All defaults can be overridden via the ``overrides`` dict.

    Example:
        >>> # Basic: just model + dataset
        >>> preset = Preset("AOSNet", "etth1")
        >>> preset.run()

        >>> # Override common params directly
        >>> preset = Preset("AOSNet", "etth1", horizon=192, lr=5e-4, epochs=20)
        >>> preset.run()

        >>> # Override any nested config via overrides dict
        >>> preset = Preset("AOSNet", "etth1",
        ...     overrides={
        ...         "model": {"individual": True},
        ...         "norm": {"affine": True},
        ...         "loss": {"loss_type": "mix_freq_mse", "alpha": 0.3},
        ...         "train": {"scheduler_type": "cosine"},
        ...         "data": {"batch_size": 64},
        ...         "wrapper": {"forecast_type": "MS"},
        ...         "trainer": {"early_stopping_patience": 5},
        ...     })
        >>> preset.run()

        >>> # Model-specific override (merged into model preset)
        >>> preset = Preset("TQNet", "etth1",
        ...     model_overrides={"period_len": 12})

        >>> # Probabilistic model
        >>> preset = Preset("NsDiff", "etth1", probabilistic=True)
        >>> preset.run()

        >>> # Inference-only mode with checkpoint (skips training)
        >>> preset = Preset("AOSNet", "etth1", ckpt="logs/version_0/checkpoints/epoch=010.ckpt")
        >>> preset.predict()
    """

    model_name: str
    dataset_name: str
    category: str = "common"
    horizon: int = 96
    lookback: int = 0  # 0 = auto from dataset defaults
    batch_size: int = 0  # 0 = auto
    lr: float = 0  # 0 = auto from model preset
    epochs: int = 0  # 0 = auto
    gpu: int = 0
    save_dir: str = "./logs"
    experiment_name: str = ""
    model_overrides: Optional[dict] = None  # Merge into model preset
    overrides: Optional[dict] = None  # Override any config section
    probabilistic: bool = False  # Use TSProbabilisticModule for probabilistic models
    ckpt: Optional[str] = None  # Path to checkpoint for inference/pretrained loading

    # Cached components
    _datamodule: Optional[DataModule] = field(default=None, repr=False)
    _model: Optional[Any] = field(default=None, repr=False)
    _lightning_module: Optional[Any] = field(default=None, repr=False)
    _trainer: Optional[TSTrainer] = field(default=None, repr=False)

    def __post_init__(self):
        """Resolve auto-defaults with overrides."""
        model_preset = _get_model_preset(self.model_name)
        ds_defaults = _get_dataset_defaults(self.dataset_name, self.horizon)

        # Apply overrides dict
        ovr = self.overrides or {}
        data_ovr = ovr.get("data", {})
        train_ovr = ovr.get("train", {})

        # Resolve with overrides
        self._resolved_lookback = data_ovr.get(
            "lookback", self.lookback if self.lookback else ds_defaults["lookback"]
        )
        self._resolved_horizon = data_ovr.get("horizon", self.horizon)
        self._resolved_batch_size = data_ovr.get(
            "batch_size",
            self.batch_size if self.batch_size else ds_defaults["batch_size"],
        )
        self._resolved_freq = data_ovr.get("freq", ds_defaults["freq"])
        self._resolved_sample_step = data_ovr.get(
            "sample_step", ds_defaults["sample_step"]
        )
        self._resolved_split_ratio = data_ovr.get("split_ratio")
        self._resolved_split_counts = data_ovr.get("split_counts")
        self._resolved_drop_last_train = data_ovr.get("drop_last_train", True)
        self._resolved_num_workers = data_ovr.get("num_workers", 0)

        self._resolved_lr = train_ovr.get(
            "lr", self.lr if self.lr else model_preset["train"]["lr"]
        )
        self._resolved_epochs = train_ovr.get(
            "epochs", self.epochs if self.epochs else model_preset["train"]["epochs"]
        )

        # Auto experiment name
        if not self.experiment_name:
            self.experiment_name = f"{self.model_name}_{self.dataset_name}"

    def _get_merged_model_config(self) -> dict:
        """Get model preset merged with model_overrides and overrides['model']."""
        model_preset = _get_model_preset(self.model_name)
        model_cfg = dict(model_preset["model"])
        # Apply overrides['model'] first
        model_cfg.update((self.overrides or {}).get("model", {}))
        # Then apply model_overrides (higher priority)
        if self.model_overrides:
            model_cfg.update(self.model_overrides)
        return model_cfg

    def _get_merged_norm_config(self) -> dict:
        """Get norm preset merged with overrides."""
        model_preset = _get_model_preset(self.model_name)
        norm_cfg = dict(model_preset["norm"])
        norm_cfg.update((self.overrides or {}).get("norm", {}))
        return norm_cfg

    def _get_merged_loss_config(self) -> dict:
        """Get loss preset merged with overrides."""
        model_preset = _get_model_preset(self.model_name)
        loss_cfg = dict(model_preset["loss"])
        loss_cfg.update((self.overrides or {}).get("loss", {}))
        return loss_cfg

    @staticmethod
    def _wrapper_loss_type(loss_cfg: dict):
        """Translate preset loss config into the wrapper's LossManager format."""
        loss_type = loss_cfg["loss_type"]
        if (
            isinstance(loss_type, str)
            and loss_type in {"mix_freq_mse", "mix_freq_mae"}
        ):
            if "alpha" in loss_cfg:
                return [loss_type, loss_cfg["alpha"]]
            if "config" in loss_cfg:
                config = loss_cfg["config"]
                return [loss_type, config[0] if isinstance(config, list) else config]
            # Backward compatibility for old experiment scripts that used
            # weight as the MixFreq alpha before alpha was exposed explicitly.
            if "weight" in loss_cfg:
                return [loss_type, loss_cfg["weight"]]
        return loss_type

    def _get_merged_train_config(self) -> dict:
        """Get train preset merged with overrides."""
        model_preset = _get_model_preset(self.model_name)
        train_cfg = dict(model_preset["train"])
        train_cfg.update((self.overrides or {}).get("train", {}))
        return train_cfg

    @property
    def datamodule(self) -> DataModule:
        """Get or create the DataModule."""
        if self._datamodule is None:
            self._datamodule = DataModule(
                category=self.category,
                dataset_name=self.dataset_name,
                lookback_len=self._resolved_lookback,
                horizon_len=self._resolved_horizon,
                batch_size=self._resolved_batch_size,
                num_workers=self._resolved_num_workers,
                freq=self._resolved_freq,
                sample_step=self._resolved_sample_step,
                split_ratio=self._resolved_split_ratio,
                split_counts=self._resolved_split_counts,
                drop_last_train=self._resolved_drop_last_train,
            )
        return self._datamodule

    @property
    def num_features(self) -> int:
        """Get number of features from datamodule."""
        return self.datamodule.num_features

    @property
    def model(self) -> Any:
        """Get or create the base model."""
        if self._model is None:
            model_kwargs = self._get_merged_model_config()
            configs = SimpleNamespace(
                num_features=self.num_features,
                lookback_len=self._resolved_lookback,
                horizon_len=self._resolved_horizon,
                **model_kwargs,
            )

            if self.probabilistic:
                from tsflib.models.diffusion import (
                    ChronosConfigs,
                    ChronosModel,
                    ConfGCConfigs,
                    ConfGCDiffConfigs,
                    ConfGCDiffModel,
                    ConfGCModel,
                    CSDIConfigs,
                    CSDIModel,
                    CycleDConfigs,
                    CycleFlowConfigs,
                    CycleFlowModel,
                    CycleJointConfigs,
                    CycleJointModel,
                    D3UConfigs,
                    D3UModel,
                    D3VAEConfigs,
                    D3VAEModel,
                    DiffusionTSConfigs,
                    DiffusionTSModel,
                    ETSConfigs,
                    ETSModel,
                    GRUMAFConfigs,
                    GRUMAFModel,
                    GRUNVPConfigs,
                    GRUNVPModel,
                    K2VAEConfigs,
                    K2VAEModel,
                    KoonproConfigs,
                    KoonproModel,
                    LagLlamaConfigs,
                    LagLlamaModel,
                    NsDiffConfigs,
                    NsDiffModel,
                    RIPCNConfigs,
                    RIPCNModel,
                    SSSDConfigs,
                    SSSDModel,
                    SundialConfigs,
                    SundialModel,
                    Tactis2Configs,
                    Tactis2Model,
                    TimeDiffConfigs,
                    TimeDiffModel,
                    TimeGradConfigs,
                    TimeGradModel,
                    TimeMCLConfigs,
                    TimeMCLModel,
                    TimePrismConfigs,
                    TimePrismModel,
                    TMDMConfigs,
                    TMDMModel,
                    TransMAFConfigs,
                    TransMAFModel,
                    TSDiffConfigs,
                    TSDiffModel,
                    TSFlowConfigs,
                    TSFlowModel,
                )
                from tsflib.models.diffusion import (
                    CycleDPretrain as CycleDModel,
                )

                prob_model_map = {
                    "NsDiff": (NsDiffModel, NsDiffConfigs),
                    "CSDI": (CSDIModel, CSDIConfigs),
                    "TimeGrad": (TimeGradModel, TimeGradConfigs),
                    "TSFlow": (TSFlowModel, TSFlowConfigs),
                    "CycleFlow": (CycleFlowModel, CycleFlowConfigs),
                    "K2VAE": (K2VAEModel, K2VAEConfigs),
                    "TimePrism": (TimePrismModel, TimePrismConfigs),
                    "D3U": (D3UModel, D3UConfigs),
                    "CycleD": (CycleDModel, CycleDConfigs),
                    "CycleJoint": (CycleJointModel, CycleJointConfigs),
                    "TSDiff": (TSDiffModel, TSDiffConfigs),
                    "TimeDiff": (TimeDiffModel, TimeDiffConfigs),
                    "SSSD": (SSSDModel, SSSDConfigs),
                    "D3VAE": (D3VAEModel, D3VAEConfigs),
                    "DiffusionTS": (DiffusionTSModel, DiffusionTSConfigs),
                    "TMDM": (TMDMModel, TMDMConfigs),
                    "Chronos": (ChronosModel, ChronosConfigs),
                    "LagLlama": (LagLlamaModel, LagLlamaConfigs),
                    "TransMAF": (TransMAFModel, TransMAFConfigs),
                    "GRUMAF": (GRUMAFModel, GRUMAFConfigs),
                    "GRUNVP": (GRUNVPModel, GRUNVPConfigs),
                    "ConfGCDiff": (ConfGCDiffModel, ConfGCDiffConfigs),
                    "ConfGC": (ConfGCModel, ConfGCConfigs),
                    "Koonpro": (KoonproModel, KoonproConfigs),
                    "RIPCN": (RIPCNModel, RIPCNConfigs),
                    "Sundial": (SundialModel, SundialConfigs),
                    "TimeMCL": (TimeMCLModel, TimeMCLConfigs),
                    "ETS": (ETSModel, ETSConfigs),
                    "Tactis2": (Tactis2Model, Tactis2Configs),
                }

                model_cls, config_cls = prob_model_map[self.model_name]
                # Create proper config dataclass for probabilistic models
                prob_config = config_cls(
                    num_features=self.num_features,
                    lookback_len=self._resolved_lookback,
                    horizon_len=self._resolved_horizon,
                    **{
                        k: v
                        for k, v in model_kwargs.items()
                        if k in config_cls.__dataclass_fields__
                    },
                )
                self._model = model_cls(prob_config)
            else:
                from tsflib.models.mlp import (
                    CycleNetModel,
                    TQNetModel,
                    AOSNetModel,
                )

                model_map = {
                    "CycleNet": CycleNetModel,
                    "TQNet": TQNetModel,
                    "AOSNet": AOSNetModel,
                }

                self._model = model_map[self.model_name](configs)
        return self._model

    @property
    def lightning_module(self) -> Any:
        """Get or create the Lightning module."""
        if self._lightning_module is None:
            norm_cfg = self._get_merged_norm_config()
            loss_cfg = self._get_merged_loss_config()
            wrapper_ovr = (self.overrides or {}).get("wrapper", {})

            wrapper_config = TSForecastingConfig(
                lr=self._resolved_lr,
                loss_type=self._wrapper_loss_type(loss_cfg),
                loss_weights=loss_cfg.get("weight"),
                use_norm=norm_cfg["use_norm"],
                norm_affine=norm_cfg["affine"],
                save_results=True,
                save_predictions=True,
                results_filename=f"{self.model_name}_{self.dataset_name}_results.json",
                **wrapper_ovr,
            )

            if self.probabilistic:
                # Get num_samples from model config
                model_kwargs = self._get_merged_model_config()
                num_samples = model_kwargs.get("num_samples", 100)

                self._lightning_module = TSProbabilisticModule(
                    model=self.model,
                    num_features=self.num_features,
                    lookback_len=self._resolved_lookback,
                    horizon_len=self._resolved_horizon,
                    num_samples=num_samples,
                    config=wrapper_config,
                )
            else:
                self._lightning_module = TSForecastingModule(
                    model=self.model,
                    num_features=self.num_features,
                    lookback_len=self._resolved_lookback,
                    horizon_len=self._resolved_horizon,
                    config=wrapper_config,
                )
        return self._lightning_module

    @property
    def trainer(self) -> TSTrainer:
        """Get or create the Trainer."""
        if self._trainer is None:
            train_cfg = self._get_merged_train_config()
            trainer_ovr = (self.overrides or {}).get("trainer", {})

            trainer_config = TSTrainerConfig(
                max_epochs=self._resolved_epochs,
                accelerator="gpu",
                devices=[self.gpu],
                save_dir=self.save_dir,
                experiment_name=self.experiment_name,
                early_stopping_patience=train_cfg["early_stopping_patience"],
                **trainer_ovr,
            )
            self._trainer = TSTrainer(trainer_config)
        return self._trainer

    def fit(self):
        """Train the model."""
        print(f"\n{'=' * 60}")
        print(f"Training: {self.model_name} @ {self.dataset_name}")
        print(f"{'=' * 60}")
        print(f"  lookback  = {self._resolved_lookback}")
        print(f"  horizon   = {self._resolved_horizon}")
        print(f"  num_feat  = {self.num_features}")
        print(f"  batch     = {self._resolved_batch_size}")
        print(f"  lr        = {self._resolved_lr}")
        print(f"  epochs    = {self._resolved_epochs}")

        param_count = sum(p.numel() for p in self.lightning_module.parameters())
        print(f"  params    = {param_count:,}")
        print(f"{'=' * 60}\n")

        torch.set_float32_matmul_precision("medium")
        self.trainer.fit(self.lightning_module, datamodule=self.datamodule)

    def _best_val_ckpt(self) -> str:
        """Get the best validation checkpoint from the trainer callback."""
        checkpoint_callback = self.trainer.checkpoint_callback
        best_ckpt = (
            getattr(checkpoint_callback, "best_model_path", None)
            if checkpoint_callback is not None
            else None
        )
        if not best_ckpt:
            raise RuntimeError(
                "Best validation checkpoint is unavailable. "
                "Run fit() first and make sure checkpointing is enabled."
            )
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")
        return best_ckpt

    def test(self, ckpt: Optional[str] = None):
        """Evaluate the model with an explicit checkpoint path."""
        if not ckpt:
            raise ValueError(
                "Preset.test() requires an explicit ckpt path. "
                "Use preset.run() to auto-test with the best validation checkpoint, "
                "or call preset.test(ckpt='...')."
            )
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        if hasattr(self.lightning_module, "set_inference_ckpt_path"):
            self.lightning_module.set_inference_ckpt_path(ckpt)

        results = self.trainer.test(
            self.lightning_module,
            datamodule=self.datamodule,
            ckpt_path=ckpt,
        )

        print(f"\n{'=' * 60}")
        print(f"Test Results: {self.model_name} @ {self.dataset_name}")
        print(f"{'=' * 60}")
        print(f"  ckpt      = {ckpt}")
        r = results[0]
        if self.probabilistic:
            # Probabilistic metrics
            for key in [
                "CRPS",
                "CRPS_sum",
                "PICP",
                "QICE",
                "ND",
                "MSE_Median",
                "MAE_Median",
            ]:
                if key in r:
                    print(f"  {key}: {r[key]:.4f}")
        else:
            # Point forecast metrics
            for key in ["MSE", "MAE", "RMSE", "MSPE", "RSE"]:
                if key in r:
                    print(f"  {key}: {r[key]:.4f}")
        print(f"{'=' * 60}\n")

        return results

    def run(self):
        """Train and test using the best validation checkpoint."""
        self.fit()
        return self.test(ckpt=self._best_val_ckpt())

    def predict(self, ckpt: Optional[str] = None):
        """Load checkpoint and run inference without training.

        Args:
            ckpt: Path to checkpoint file. If None, uses self.ckpt from constructor.

        Returns:
            Test results dict with metrics and saved predictions.
        """
        ckpt_path = ckpt or self.ckpt
        if ckpt_path is None:
            raise ValueError(
                "Checkpoint path required for inference. "
                "Provide via preset.ckpt='path/to.ckpt' or preset.predict(ckpt='path')"
            )

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"\n{'=' * 60}")
        print(f"Inference: {self.model_name} @ {self.dataset_name}")
        print(f"{'=' * 60}")
        print(f"  lookback  = {self._resolved_lookback}")
        print(f"  horizon   = {self._resolved_horizon}")
        print(f"  num_feat  = {self.num_features}")
        print(f"  batch     = {self._resolved_batch_size}")
        print(f"  ckpt      = {ckpt_path}")
        print(f"{'=' * 60}\n")

        torch.set_float32_matmul_precision("medium")

        if hasattr(self.lightning_module, "set_inference_ckpt_path"):
            self.lightning_module.set_inference_ckpt_path(ckpt_path)

        # Use trainer.test with ckpt_path to load weights and run inference
        results = self.trainer.test(
            self.lightning_module, datamodule=self.datamodule, ckpt_path=ckpt_path
        )

        print(f"\n{'=' * 60}")
        print(f"Inference Results: {self.model_name} @ {self.dataset_name}")
        print(f"{'=' * 60}")
        r = results[0]
        if self.probabilistic:
            for key in [
                "CRPS",
                "CRPS_sum",
                "PICP",
                "QICE",
                "ND",
                "MSE_Median",
                "MAE_Median",
            ]:
                if key in r:
                    print(f"  {key}: {r[key]:.4f}")
        else:
            for key in ["MSE", "MAE", "RMSE", "MSPE", "RSE"]:
                if key in r:
                    print(f"  {key}: {r[key]:.4f}")
        print(f"{'=' * 60}\n")
        # else:
        #     for key in ["MSE", "MAE", "RMSE", "MSPE", "RSE"]:
        #         if key in r:
        #             print(f"  {key}: {r[key]:.4f}")
        # print(f"{'='*60}\n")

        return results

    def _get_version_dir(self) -> str:
        """Get the latest version directory path (e.g., logs/AOSNet_etth1/version_1)."""
        log_dir = f"{self.save_dir}/{self.experiment_name}"
        import glob

        versions = sorted(glob.glob(f"{log_dir}/version_*"))
        if versions:
            return versions[-1]
        return log_dir

    def _get_predictions_path(self) -> str:
        """Get the path to saved predictions.npz."""
        version_dir = self._get_version_dir()
        return os.path.join(version_dir, "predictions.npz")

    def _load_predictions(self) -> dict:
        """Load predictions from saved predictions.npz.

        For point forecast models:
            Returns dict with keys: predictions, targets, inputs.
            Shapes: predictions (N, C, H), targets (N, C, H), inputs (N, C, L).

        For probabilistic models:
            Returns dict with keys: samples, targets, inputs, median, mean, etc.
            Shapes: samples (N, num_samples, H, C), targets (N, H, C), inputs (N, L, C).
        """
        npz_path = self._get_predictions_path()
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Predictions not found at {npz_path}. "
                "Run preset.test() first to generate predictions."
            )

        data = np.load(npz_path)

        if self.probabilistic:
            # Probabilistic models save samples, targets, inputs, median, mean, etc.
            result = {
                "samples": data["samples"],
                "targets": data["targets"],
            }
            if "inputs" in data:
                result["inputs"] = data["inputs"]
            if "median" in data:
                result["median"] = data["median"]
            if "mean" in data:
                result["mean"] = data["mean"]
            if "lower_05" in data:
                result["lower_05"] = data["lower_05"]
            if "upper_95" in data:
                result["upper_95"] = data["upper_95"]
        else:
            # Point forecast models save predictions, targets, inputs
            result = {
                "predictions": data["predictions"],
                "targets": data["targets"],
            }
            if "inputs" in data:
                result["inputs"] = data["inputs"]
        return result

    def plot(
        self,
        sample_id: int = 0,
        channel: int = 0,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot single-channel forecast with lookback.

        Loads predictions from the most recent test run.

        Args:
            sample_id: Sample index to plot.
            channel: Feature/channel index to plot.
            save_path: Output path. Defaults to ``{version_dir}/plot.png``.

        Returns:
            Path to the saved plot file.
        """
        if self.probabilistic:
            raise ValueError(
                f"plot() is for point forecast models. "
                f"Use plot_prob() for probabilistic models like {self.model_name}."
            )
        results = self._load_predictions()

        if save_path is None:
            version_dir = self._get_version_dir()
            save_path = os.path.join(version_dir, "plot.png")

        plot_point_forecast(
            predictions=results["predictions"],
            targets=results["targets"],
            inputs=results.get("inputs"),
            sample_id=sample_id,
            channel=channel,
            lookback_len=self._resolved_lookback,
            save_path=save_path,
        )
        return save_path

    def plot_multi(
        self,
        sample_id: int = 0,
        channels: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot multi-channel forecast with lookback.

        Loads predictions from the most recent test run.

        Args:
            sample_id: Sample index to plot.
            channels: List of channel indices. Defaults to first 4.
            save_path: Output path. Defaults to ``{version_dir}/multi.png``.

        Returns:
            Path to the saved plot file.
        """
        if self.probabilistic:
            raise ValueError(
                f"plot_multi() is for point forecast models. "
                f"Use plot_prob_multi() for probabilistic models like {self.model_name}."
            )
        results = self._load_predictions()

        if save_path is None:
            version_dir = self._get_version_dir()
            save_path = os.path.join(version_dir, "multi.png")

        plot_point_multi_channel(
            predictions=results["predictions"],
            targets=results["targets"],
            inputs=results.get("inputs"),
            sample_id=sample_id,
            channels=channels,
            lookback_len=self._resolved_lookback,
            save_path=save_path,
        )
        return save_path

    def plot_prob(
        self,
        sample_id: int = 0,
        channel: int = 0,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot single-channel probabilistic forecast with confidence intervals.

        Loads predictions from the most recent test run.

        Args:
            sample_id: Sample index to plot.
            channel: Feature/channel index to plot.
            save_path: Output path. Defaults to ``{version_dir}/prob.png``.

        Returns:
            Path to the saved plot file.
        """
        if not self.probabilistic:
            raise ValueError(
                "plot_prob() requires probabilistic=True. Use plot() for point forecasts."
            )

        npz_path = self._get_predictions_path()
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Predictions not found at {npz_path}. "
                "Run preset.test() first to generate predictions."
            )

        data = np.load(npz_path)
        samples = data["samples"]  # (N, num_samples, H, C)
        targets = data["targets"]  # (N, H, C)
        inputs = data.get("inputs", None)  # (N, L, C)

        if save_path is None:
            version_dir = self._get_version_dir()
            save_path = os.path.join(version_dir, "prob.png")

        plot_prob_forecast(
            samples=samples,
            targets=targets,
            inputs=inputs,
            sample_id=sample_id,
            channel=channel,
            lookback_len=self._resolved_lookback,
            save_path=save_path,
        )
        return save_path

    def plot_prob_multi(
        self,
        sample_id: int = 0,
        channels: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot multi-channel probabilistic forecast with confidence intervals.

        Loads predictions from the most recent test run.

        Args:
            sample_id: Sample index to plot.
            channels: List of channel indices. Defaults to first 4.
            save_path: Output path. Defaults to ``{version_dir}/prob_multi.png``.

        Returns:
            Path to the saved plot file.
        """
        if not self.probabilistic:
            raise ValueError(
                "plot_prob_multi() requires probabilistic=True. Use plot_multi() for point forecasts."
            )

        npz_path = self._get_predictions_path()
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Predictions not found at {npz_path}. "
                "Run preset.test() first to generate predictions."
            )

        data = np.load(npz_path)
        samples = data["samples"]  # (N, num_samples, H, C)
        targets = data["targets"]  # (N, H, C)
        inputs = data.get("inputs", None)  # (N, L, C)

        if save_path is None:
            version_dir = self._get_version_dir()
            save_path = os.path.join(version_dir, "prob_multi.png")

        plot_prob_multi_channel(
            samples=samples,
            targets=targets,
            inputs=inputs,
            sample_id=sample_id,
            channels=channels,
            lookback_len=self._resolved_lookback,
            save_path=save_path,
        )
        return save_path


# --------------------------------------------------------------------------
# Top-level convenience function
# --------------------------------------------------------------------------


def train(
    model_name: str,
    dataset_name: str,
    category: str = "common",
    horizon: int = 96,
    lookback: int = 0,
    batch_size: int = 0,
    lr: float = 0,
    epochs: int = 10,
    gpu: int = 0,
    save_dir: str = "./logs",
    probabilistic: bool = False,
    model_overrides: Optional[dict] = None,
    overrides: Optional[dict] = None,
    ckpt: Optional[str] = None,
):
    """One-line training and testing.

    Args:
        model_name: Model name (e.g., 'AOSNet', 'TQNet', 'NsDiff').
        dataset_name: Dataset name (e.g., 'etth1', 'weather').
        category: Data category (default 'common').
        horizon: Forecast horizon (default 96).
        lookback: Input length (0 = auto).
        batch_size: Batch size (0 = auto).
        lr: Learning rate (0 = auto).
        epochs: Training epochs (0 = auto).
        gpu: GPU index.
        save_dir: Log directory.
        probabilistic: Use TSProbabilisticModule for probabilistic models (e.g., NsDiff).
        model_overrides: Override model-specific params (e.g., {'period_len': 12}).
        overrides: Override any config section (model/norm/loss/train/data/wrapper/trainer).
        ckpt: Path to checkpoint for inference-only mode (skips training).

    Example:
        >>> from tsflib import train
        >>> train("AOSNet", "etth1")
        >>> train("TQNet", "etth1", horizon=192)
        >>> train("NsDiff", "etth1", probabilistic=True)
        >>> train("AOSNet", "etth1", overrides={
        ...     "model": {"individual": True},
        ...     "norm": {"affine": True},
        ...     "loss": {"loss_type": "mix_freq_mse", "alpha": 0.3},
        ... })
        >>> # Inference-only mode with checkpoint
        >>> train("AOSNet", "etth1", ckpt="logs/AOSNet_etth1/version_0/checkpoints/epoch=010.ckpt")
    """
    preset = Preset(
        model_name=model_name,
        dataset_name=dataset_name,
        category=category,
        horizon=horizon,
        lookback=lookback,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        gpu=gpu,
        save_dir=save_dir,
        probabilistic=probabilistic,
        model_overrides=model_overrides,
        overrides=overrides,
        ckpt=ckpt,
    )

    if ckpt is not None:
        return preset.predict(ckpt=ckpt)
    return preset.run()


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------

__all__ = [
    "Preset",
    "train",
    "list_models",
    "list_datasets",
]


def list_models() -> list:
    """List all available model names."""
    return sorted(_MODEL_PRESETS.keys())


def list_datasets() -> list:
    """List all available dataset names."""
    return sorted(_DATASET_DEFAULTS.keys())
