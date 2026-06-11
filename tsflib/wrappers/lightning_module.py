"""
PyTorch Lightning Module Wrappers for Time Series Forecasting

This module provides LightningModule implementations that wrap time series
forecasting models for use with PyTorch Lightning's training infrastructure.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tsflib.losses import get_loss, LossManager
from tsflib.metrics import (
    POINT_METRICS,
    PROBABILISTIC_METRICS,
    compute_point_metrics,
    compute_prob_metrics,
    mse_torch,
    mae_torch,
)
from tsflib.modules import RevIN


@dataclass
class TSForecastingConfig:
    """Configuration for TSForecastingModule.

    Attributes:
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        loss_type: Loss function specification. Can be:
            - ``str``: Single loss name, e.g. ``"mse"``
            - ``List``: ``[loss_name, alpha]`` for single loss with config
            - ``List[List]``: ``[[name, weight], ...]`` for multi-loss
        loss_weights: Weights for multi-loss mode.
        use_norm: Whether to use normalization.
        norm_type: Normalization type ('revin', etc.).
        norm_affine: Whether to use affine in normalization.
        norm_time_dim: Index of the time dimension for normalization. Defaults to -1.
        norm_feature_dim: Index of the feature dimension for normalization. Defaults to 1.
        scheduler_type: LR scheduler type ('plateau', 'step', 'cosine', 'onecycle').
        scheduler_patience: Patience for LR scheduler.
        scheduler_factor: Factor for LR scheduler.
        onecycle_total_steps: Optional explicit total steps for OneCycleLR.
            If omitted, Lightning's estimated_stepping_batches is used.
        onecycle_pct_start: Percentage of cycle spent increasing the learning rate.
        onecycle_div_factor: Initial LR divisor for OneCycleLR.
        onecycle_final_div_factor: Final LR divisor for OneCycleLR.
        onecycle_three_phase: Whether to use three-phase OneCycleLR.
        forecast_type: Forecasting type ('M', 'MS', 'S', 'SS').
        save_results: Whether to save test results to file.
        save_predictions: Whether to save predictions and targets for visualization.
        results_dir: Directory to save results.
        results_filename: Filename for results.
        metric_names: Optional list of metric names to compute. Defaults to all
            point metrics for ``TSForecastingModule`` and all probabilistic
            metrics for ``TSProbabilisticModule``.
    """

    lr: float = 1e-3
    weight_decay: float = 0.0
    loss_type: Any = "mse"
    loss_weights: Optional[List[float]] = None
    use_norm: bool = True
    norm_type: str = "revin"
    norm_affine: bool = False
    norm_time_dim: int = -1
    norm_feature_dim: int = 1
    scheduler_type: str = "plateau"
    scheduler_patience: int = 1
    scheduler_factor: float = 0.3
    onecycle_total_steps: Optional[int] = None
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    onecycle_three_phase: bool = False
    forecast_type: str = "M"
    save_results: bool = True
    save_predictions: bool = False
    results_dir: Optional[str] = None
    results_filename: str = "test_results.json"
    inference_ckpt_path: Optional[str] = None
    model_name: Optional[str] = None
    dataset: Optional[str] = None
    metric_names: Optional[List[str]] = None
    profiling: Optional[List[str]] = (
        None  # e.g. ['epoch_time', 'num_params', 'flops', 'max_gpu_memory', 'inference_time_per_batch', 'all']
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class TSForecastingModule(L.LightningModule):
    """PyTorch Lightning Module for time series forecasting.

    This class wraps a time series forecasting model and provides
    training, validation, and testing logic compatible with PyTorch Lightning.

    Args:
        model: The forecasting model to wrap.
        num_features: Number of input features.
        lookback_len: Length of input sequence.
        horizon_len: Length of forecast horizon.
        config: Configuration object.
        lr: Learning rate (overrides config if provided).
        loss_type: Loss type (overrides config if provided).

    Example:
        >>> from tsflib.models import DLinear
        >>> from tsflib.wrappers import TSForecastingModule
        >>>
        >>> # Create base model
        >>> configs = DLinear.Configs(
        ...     num_features=7,
        ...     lookback_len=96,
        ...     horizon_len=96
        ... )
        >>> base_model = DLinear.Model(configs)
        >>>
        >>> # Wrap for Lightning
        >>> pl_module = TSForecastingModule(
        ...     model=base_model,
        ...     num_features=7,
        ...     lookback_len=96,
        ...     horizon_len=96,
        ...     lr=1e-3,
        ...     loss_type='mse'
        ... )
        >>>
        >>> # Train
        >>> trainer.fit(pl_module, datamodule)
    """

    def __init__(
        self,
        model: nn.Module,
        num_features: Optional[int] = None,
        lookback_len: Optional[int] = None,
        horizon_len: Optional[int] = None,
        config: Optional[TSForecastingConfig] = None,
        lr: Optional[float] = None,
        loss_type: Optional[str] = None,
        **kwargs,
    ):
        # Infer dimensions from model.configs if not explicitly provided
        model_cfg = getattr(model, "configs", None)
        if num_features is None:
            num_features = getattr(model_cfg, "num_features", None)
            if num_features is None:
                num_features = getattr(model, "num_features", None)
            if num_features is None:
                num_features = getattr(model, "enc_in", None)
        if lookback_len is None:
            lookback_len = getattr(model_cfg, "lookback_len", None)
            if lookback_len is None:
                lookback_len = getattr(model, "lookback_len", None)
            if lookback_len is None:
                lookback_len = getattr(model, "seq_len", None)
        if horizon_len is None:
            horizon_len = getattr(model_cfg, "horizon_len", None)
            if horizon_len is None:
                horizon_len = getattr(model, "horizon_len", None)
            if horizon_len is None:
                horizon_len = getattr(model, "pred_len", None)

        if num_features is None:
            raise ValueError(
                "num_features could not be inferred from model. " "Please pass it explicitly."
            )
        if lookback_len is None:
            raise ValueError(
                "lookback_len could not be inferred from model. " "Please pass it explicitly."
            )
        if horizon_len is None:
            raise ValueError(
                "horizon_len could not be inferred from model. " "Please pass it explicitly."
            )

        # Extract model-specific hyperparams from model.configs
        # (e.g. period, d_model, individual, hidden_size, n_heads, ...)
        model_configs = {}
        # Scan the model's own relevant attributes
        known_model_attrs = [
            "individual", "period", "d_model", "num_selected", "hidden_size",
            "dropout", "lambda_global", "beta_ema", "gamma_contrast",
            "use_contrast", "use_adjacent", "patch_len", "stride",
            "n_heads", "enc_in", "seq_len", "pred_len",
        ]
        for attr in known_model_attrs:
            if hasattr(model, attr) and not callable(getattr(model, attr)):
                val = getattr(model, attr, None)
                if val is not None and attr not in ("num_features", "lookback_len", "horizon_len"):
                    model_configs[attr] = val

        super().__init__()
        # Exclude the dimension args so they don't conflict with DataModule
        # hparams of the same name.  We store them as plain attributes instead.
        self.save_hyperparameters(ignore=["model", "num_features", "lookback_len", "horizon_len", "model_configs"])
        # Save model-specific configs as a separate hparam namespace
        if model_configs:
            self.hparams["model_configs"] = model_configs

        # Store model
        self.model = model

        # Configuration
        self.config = config or TSForecastingConfig()

        # Override config with direct parameters
        if lr is not None:
            self.config.lr = lr
        if loss_type is not None:
            self.config.loss_type = loss_type

        # Store dimensions as plain attributes (not in hparams)
        self.num_features = num_features
        self.lookback_len = lookback_len
        self.horizon_len = horizon_len

        # Setup normalization
        self.normalizer = self._setup_normalization()

        # Setup loss function (supports single or multi-loss via LossManager)
        self.criterion = LossManager(
            loss_type=self.config.loss_type,
            loss_weights=self.config.loss_weights,
        )

        # Storage for outputs
        self.train_outputs: List[Dict[str, torch.Tensor]] = []
        self.val_outputs: List[Dict[str, torch.Tensor]] = []
        self.test_outputs: List[Dict[str, torch.Tensor]] = []

    def set_inference_ckpt_path(self, ckpt_path: Optional[str]) -> None:
        """Record which checkpoint is used for test/predict inference."""
        self.config.inference_ckpt_path = ckpt_path

    def _setup_normalization(self):
        """Setup normalization layer."""
        return RevIN(
            use_norm=self.config.use_norm if hasattr(self.config, "use_norm") else True,
            num_features=self.num_features,
            time_dim=self.config.norm_time_dim if hasattr(self.config, "norm_time_dim") else -1,
            feature_dim=self.config.norm_feature_dim if hasattr(self.config, "norm_feature_dim") else 1,
            affine=self.config.norm_affine if hasattr(self.config, "norm_affine") else False,
            subtract_last=self.config.subtract_last if hasattr(self.config, "subtract_last") else False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, L).

        Returns:
            Output tensor of shape (B, C, H).
        """
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return self.model(x)

    def _process_batch(
        self, batch: Tuple, mode: str = "train"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a batch.

        The entire batch tuple is forwarded to the model so that each model
        can pick the fields it needs internally (e.g. CycleNet extracts
        ``batch[4] % cycle_len`` for cycle_index).

        Args:
            batch: Batch tuple (x, y) or (x, y, x_mark, y_mark, index).
            mode: Mode ('train', 'val', 'test').

        Returns:
            Tuple of (pred, target, loss).
        """
        # Handle different batch formats
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 5:
            x, y, x_mark, y_mark, index = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        x = x.to(torch.float32)
        y = y.to(torch.float32)

        # Normalize input
        x = self.normalizer(x, mode="norm")

        # Forward pass — convert batch to float32 and pass to model;
        # each model decides internally which fields to use.
        if isinstance(batch, (tuple, list)):
            batch_items = list(batch)
            batch_items[0] = x
            for i in range(1, len(batch_items)):
                if isinstance(batch_items[i], torch.Tensor):
                    batch_items[i] = batch_items[i].to(torch.float32)
            batch_float32 = tuple(batch_items)
        else:
            batch_float32 = x

        pred = self.model(batch_float32)

        # Denormalize output
        pred = self.normalizer(pred, mode="denorm")

        # Extract relevant dimensions based on forecast type
        if self.config.forecast_type in ("MS", "SS"):
            # Single target
            pred = pred[:, -1:, -self.horizon_len :]
            target = y[:, -1:, -self.horizon_len :]
        else:
            # All features
            pred = pred[:, :, -self.horizon_len :]
            target = y[:, :, -self.horizon_len :]

        # Compute loss
        loss = self.criterion(pred, target)

        # Add auxiliary loss from model (e.g., contrastive loss in PRISM)
        if hasattr(self.model, "_aux_loss") and isinstance(
            self.model._aux_loss, torch.Tensor
        ):
            loss = loss + self.model._aux_loss

        return pred, target, loss

    def training_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        pred, target, loss = self._process_batch(batch, mode="train")

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.train_outputs.append({"loss": loss})

        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        if not self.train_outputs:
            return

        avg_loss = torch.stack([o["loss"] for o in self.train_outputs]).mean()
        self.log("train_epoch_loss", avg_loss, prog_bar=True, sync_dist=True)

        self.train_outputs.clear()

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        pred, target, loss = self._process_batch(batch, mode="val")

        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_mse",
            mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_mae",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.val_outputs.append({"loss": loss, "mse": mse, "mae": mae})

        return {"loss": loss, "mse": mse, "mae": mae}

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if not self.val_outputs:
            return

        avg_loss = torch.stack([o["loss"] for o in self.val_outputs]).mean()
        avg_mse = torch.stack([o["mse"] for o in self.val_outputs]).mean()
        avg_mae = torch.stack([o["mae"] for o in self.val_outputs]).mean()

        self.log("val_epoch_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_epoch_mse", avg_mse, prog_bar=True, sync_dist=True)
        self.log("val_epoch_mae", avg_mae, prog_bar=True, sync_dist=True)

        self.val_outputs.clear()

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        # Save raw lookback input before _process_batch modifies it
        raw_input = batch[0].float().cpu()

        pred, target, loss = self._process_batch(batch, mode="test")

        output = {
            "pred": pred.cpu(),
            "target": target.cpu(),
        }
        if self.config.save_predictions:
            output["input"] = raw_input
        self.test_outputs.append(output)

        return {"pred": pred, "target": target}

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        if not self.test_outputs:
            return

        # Aggregate predictions
        all_preds = torch.cat([o["pred"] for o in self.test_outputs], dim=0)
        all_targets = torch.cat([o["target"] for o in self.test_outputs], dim=0)

        # Compute metrics using numpy for consistency with evaluator
        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()

        # Save predictions and targets before metric computation so inference
        # outputs survive slow or interrupted metric evaluation.
        if self.config.save_predictions:
            all_inputs = torch.cat([o["input"] for o in self.test_outputs], dim=0)
            self._save_predictions(preds_np, targets_np, all_inputs.numpy())

        metrics = compute_point_metrics(
            targets_np,
            preds_np,
            metric_names=self.config.metric_names,
        )

        # Log metrics
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True, sync_dist=True)

        # Print summary
        print(f"\nTest Results:")
        for name in POINT_METRICS:
            if name in metrics:
                print(f"  {name}: {metrics[name]:.4f}")

        # Save results if enabled
        if self.config.save_results:
            self._save_test_results(metrics)

        # Clear outputs
        self.test_outputs.clear()

    def _get_save_dir(self) -> str:
        """Get the directory to save results."""
        if self.config.results_dir:
            save_dir = self.config.results_dir
        elif self.logger and hasattr(self.logger, "log_dir"):
            save_dir = self.logger.log_dir
        else:
            save_dir = "./results"
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _save_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        inputs: np.ndarray = None,
    ) -> None:
        """Save predictions, targets and lookback inputs as .npz for visualization.

        Args:
            predictions: Array of shape (N, C, H).
            targets: Array of shape (N, C, H).
            inputs: Array of shape (N, C, L), lookback inputs.
        """
        save_dir = self._get_save_dir()
        filepath = os.path.join(save_dir, "predictions.npz")

        save_dict = {
            "predictions": predictions,
            "targets": targets,
        }
        if inputs is not None:
            save_dict["inputs"] = inputs

        np.savez(filepath, **save_dict)

        print(f"Predictions saved to: {filepath}")
        print(f"  predictions shape: {predictions.shape}")
        print(f"  targets shape: {targets.shape}")
        if inputs is not None:
            print(f"  inputs shape: {inputs.shape}  (lookback)")

    @staticmethod
    def _infer_model_name(model: nn.Module) -> str:
        """Infer model name from its module path.

        E.g. tsflib.models.diffusion.K2VAE -> K2VAE
        """
        module = getattr(model, "__module__", "")
        # Use the last segment of the module path (e.g. "K2VAE")
        parts = [p for p in module.split(".") if p[0].isupper()]
        if parts:
            return parts[-1]
        # Fallback: class name (often just "Model")
        cls_name = type(model).__name__
        return cls_name if cls_name != "Model" else module.split(".")[-1]

    @staticmethod
    def _infer_dataset_name(trainer) -> str:
        """Infer dataset name from the trainer's datamodule."""
        dm = getattr(trainer, "datamodule", None)
        if dm is None:
            return ""
        # DataModule has dataset_name when loaded from TSDATADIR.
        name = getattr(dm, "dataset_name", None)
        if name:
            return name
        # DataModule: extract stem from data_path
        data_path = getattr(dm, "data_path", "")
        if data_path:
            return os.path.splitext(os.path.basename(data_path))[0]
        return ""

    def _collect_profiling(self) -> Dict[str, Any]:
        """Collect profiling results from the ProfilingCallback (if any)."""
        if self.trainer is None:
            return {}
        for cb in self.trainer.callbacks:
            from tsflib.trainers.callbacks import ProfilingCallback

            if isinstance(cb, ProfilingCallback):
                return dict(cb.results)
        return {}

    def _save_test_results(self, results: Dict[str, Any]) -> None:
        """Save test results to file."""
        save_dir = self._get_save_dir()

        # Auto-infer model_name and dataset
        model_name = self.config.model_name or self._infer_model_name(self.model)
        dataset = self.config.dataset or self._infer_dataset_name(self.trainer)

        # Prepare results dictionary
        results_dict: Dict[str, Any] = {
            "model_name": model_name,
            "dataset": dataset,
            "metrics": results,
        }

        # Profiling info
        profiling = self._collect_profiling()
        if profiling:
            results_dict["profiling"] = profiling

        results_dict["config"] = self.config.to_dict() if hasattr(self.config, "to_dict") else {}

        # Save loss alpha if the loss function has one
        if hasattr(self.criterion, "_losses"):
            for loss_fn in self.criterion._losses:
                if hasattr(loss_fn, "alpha"):
                    results_dict["config"]["loss_alpha"] = loss_fn.alpha
                    break
        results_dict["model_info"] = {
            "num_features": self.num_features,
            "lookback_len": self.lookback_len,
            "horizon_len": self.horizon_len,
        }

        # Model-specific hyperparams (e.g. individual, period, d_model)
        model_configs = getattr(self, "hparams", {}).get("model_configs", {})
        if model_configs:
            results_dict["model_configs"] = model_configs

        # Save to JSON
        filepath = os.path.join(save_dir, self.config.results_filename)
        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        if self.config.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        elif self.config.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.scheduler_patience * 2,
                gamma=self.config.scheduler_factor,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 100,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.scheduler_type == "onecycle":
            total_steps = self.config.onecycle_total_steps
            if total_steps is None:
                try:
                    total_steps = int(self.trainer.estimated_stepping_batches)
                except RuntimeError as exc:
                    raise ValueError(
                        "OneCycleLR requires onecycle_total_steps when the module "
                        "is not attached to a Trainer."
                    ) from exc
            if total_steps <= 0:
                raise ValueError("OneCycleLR requires a positive total_steps value.")

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.lr,
                total_steps=total_steps,
                pct_start=self.config.onecycle_pct_start,
                div_factor=self.config.onecycle_div_factor,
                final_div_factor=self.config.onecycle_final_div_factor,
                three_phase=self.config.onecycle_three_phase,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def predict_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Prediction step."""
        x = (
            batch[0].to(torch.float32)
            if isinstance(batch, (tuple, list))
            else batch.to(torch.float32)
        )

        x = self.normalizer(x, mode="norm")

        with torch.no_grad():
            if isinstance(batch, (tuple, list)):
                batch_items = list(batch)
                batch_items[0] = x
                for i in range(1, len(batch_items)):
                    if isinstance(batch_items[i], torch.Tensor):
                        batch_items[i] = batch_items[i].to(torch.float32)
                pred = self.model(tuple(batch_items))
            else:
                pred = self.model(x)

        pred = self.normalizer(pred, mode="denorm")

        return pred


class TSProbabilisticModule(TSForecastingModule):
    """Lightning Module for probabilistic time series forecasting.

    This module extends TSForecastingModule for models that output
    distributions rather than point estimates (e.g., CSDI, TimeGrad).

    Metrics computed at test time:
        - CRPS: Continuous Ranked Probability Score
        - CRPS_sum: CRPS on sum of dimensions
        - PICP: Prediction Interval Coverage Probability (90%)
        - QICE: Quantile Interval Coverage Error
        - ND (NMAE): Normalized Deviation
        - MSE_Median: MSE using median forecast
        - MAE_Median: MAE using median forecast

    Args:
        model: Probabilistic forecasting model.
        num_features: Number of input features.
        lookback_len: Length of input sequence.
        horizon_len: Length of forecast horizon.
        num_samples: Number of samples for prediction.
        config: Configuration object.

    Example:
        >>> from tsflib.models.diffusion import CSDI
        >>>
        >>> configs = CSDI.CSDIConfigs(num_features=7, lookback_len=96, horizon_len=96)
        >>> model = CSDI.Model(configs)
        >>> pl_module = TSProbabilisticModule(
        ...     model=model,
        ...     num_features=7,
        ...     lookback_len=96,
        ...     horizon_len=96,
        ...     num_samples=100
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        num_features: Optional[int] = None,
        lookback_len: Optional[int] = None,
        horizon_len: Optional[int] = None,
        num_samples: int = 100,
        config: Optional[TSForecastingConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = TSForecastingConfig()
        super().__init__(
            model=model,
            num_features=num_features,
            lookback_len=lookback_len,
            horizon_len=horizon_len,
            config=config,
            **kwargs,
        )
        self.num_samples = num_samples
        self.test_outputs: List[Dict[str, torch.Tensor]] = []

    def training_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step for probabilistic models."""
        if hasattr(self.model, "train_loss"):
            loss = self.model.train_loss(batch)
        else:
            _, _, loss = self._process_batch(batch, mode="train")

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": loss}

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step for probabilistic models."""
        if hasattr(self.model, "val_loss"):
            loss = self.model.val_loss(batch)
        else:
            _, _, loss = self._process_batch(batch, mode="val")

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": loss}

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step for probabilistic models.

        Generates samples and stores them for metric computation.
        """
        # Save raw lookback input before model inference
        raw_input = batch[0].float().cpu()

        with torch.no_grad():
            if hasattr(self.model, "sample"):
                samples = self.model.sample(batch, num_samples=self.num_samples)
            else:
                samples = self.model(batch)

        # Get targets
        if len(batch) >= 2:
            targets = batch[1]
        else:
            targets = batch[0]

        # Store for epoch-end metrics
        output = {
            "samples": samples.cpu(),
            "targets": targets.cpu(),
        }
        if self.config.save_predictions:
            output["input"] = raw_input
        self.test_outputs.append(output)

        return {"samples": samples, "targets": targets}

    def on_test_epoch_end(self) -> None:
        """Compute probabilistic metrics at the end of testing."""
        if not self.test_outputs:
            return

        # Aggregate all samples and targets
        all_samples = torch.cat([o["samples"] for o in self.test_outputs], dim=0)
        all_targets = torch.cat([o["targets"] for o in self.test_outputs], dim=0)

        # Convert to numpy
        # Expected shapes:
        #   samples: (N, num_samples, C, H) -> transpose to (N, num_samples, H, C)
        #   targets: (N, C, H) -> transpose to (N, H, C)
        samples_np = all_samples.numpy()
        targets_np = all_targets.numpy()

        # Normalize shapes for metric computation: (N, S, H, C) and (N, H, C)
        if samples_np.ndim == 4:
            # (N, S, C, H) -> (N, S, H, C)
            if samples_np.shape[2] == self.num_features and samples_np.shape[3] == self.horizon_len:
                samples_np = samples_np.transpose(0, 1, 3, 2)
        if targets_np.ndim == 3:
            # (N, C, H) -> (N, H, C)
            if targets_np.shape[1] == self.num_features and targets_np.shape[2] == self.horizon_len:
                targets_np = targets_np.transpose(0, 2, 1)

        # Save predictions and targets before metric computation so inference
        # outputs survive slow or interrupted metric evaluation.
        if self.config.save_predictions:
            all_inputs = torch.cat([o["input"] for o in self.test_outputs], dim=0)
            inputs_np = all_inputs.numpy()
            # (N, C, L) -> (N, L, C) to match prob convention
            if inputs_np.ndim == 3:
                if (
                    inputs_np.shape[1] == self.num_features
                    and inputs_np.shape[2] == self.lookback_len
                ):
                    inputs_np = inputs_np.transpose(0, 2, 1)
            self._save_prob_predictions(samples_np, targets_np, inputs_np)

        # Compute probabilistic metrics
        metrics = compute_prob_metrics(
            targets_np,
            samples_np,
            metric_names=self.config.metric_names,
        )

        # Log metrics
        for name, value in metrics.items():
            self.log(f"test_{name}", value, prog_bar=True, sync_dist=True)

        # Print summary
        print("\nProbabilistic Test Results:")
        for name in PROBABILISTIC_METRICS:
            if name in metrics:
                print(f"  {name}: {metrics[name]:.4f}")

        # Save results
        if self.config.save_results:
            self._save_test_results(metrics)

        # Clear outputs
        self.test_outputs.clear()

    def _save_prob_predictions(
        self,
        samples: np.ndarray,
        targets: np.ndarray,
        inputs: np.ndarray = None,
    ) -> None:
        """Save probabilistic predictions and targets as .npz for visualization.

        Args:
            samples: Forecast samples, shape (N, num_samples, H, C).
            targets: Ground truth, shape (N, H, C).
            inputs: Lookback inputs, shape (N, L, C).
        """
        save_dir = self._get_save_dir()
        filepath = os.path.join(save_dir, "predictions.npz")

        median = np.quantile(samples, 0.5, axis=1)
        mean = samples.mean(axis=1)
        lower = np.quantile(samples, 0.05, axis=1)
        upper = np.quantile(samples, 0.95, axis=1)

        save_dict = {
            "samples": samples,
            "targets": targets,
            "median": median,
            "mean": mean,
            "lower_05": lower,
            "upper_95": upper,
        }
        if inputs is not None:
            save_dict["inputs"] = inputs

        np.savez(filepath, **save_dict)

        print(f"Predictions saved to: {filepath}")
        print(f"  samples shape: {samples.shape}  (N, num_samples, H, C)")
        print(f"  targets shape: {targets.shape}  (N, H, C)")
        if inputs is not None:
            print(f"  inputs shape: {inputs.shape}  (N, L, C)  (lookback)")

    def predict_step(
        self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Generate probabilistic predictions."""
        with torch.no_grad():
            if hasattr(self.model, "sample"):
                samples = self.model.sample(batch, num_samples=self.num_samples)
            else:
                samples = self.model(batch)

        mean = samples.mean(dim=1)
        std = samples.std(dim=1)

        return {
            "mean": mean,
            "std": std,
            "samples": samples,
        }


__all__ = [
    "TSForecastingConfig",
    "TSForecastingModule",
    "TSProbabilisticModule",
]
