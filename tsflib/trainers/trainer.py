"""
TSFLib Trainer Implementation

This module provides a high-level trainer interface for time series forecasting
models, built on top of PyTorch Lightning.
"""

import os

# Disable Rich in Jupyter notebooks to avoid recursion error
# Must be set before Rich is imported
if "RICH_ENABLED" not in os.environ:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            os.environ["RICH_ENABLED"] = "0"
    except NameError:
        pass

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import lightning as L
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    TQDMProgressBar,
    Timer,
)
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger

from tsflib.trainers.callbacks import MetricsCallback, ProfilingCallback, TrainingTimer, EpochMetricsPrinter


def _is_notebook() -> bool:
    """Detect if running in a Jupyter notebook environment."""
    return os.environ.get("RICH_ENABLED") == "0"


@dataclass
class TSTrainerConfig:
    """Configuration for TSTrainer.

    Attributes:
        max_epochs: Maximum number of training epochs.
        min_epochs: Minimum number of training epochs.
        accelerator: Accelerator type ('cpu', 'gpu', 'tpu', etc.).
        devices: Number of devices to use.
        precision: Training precision (16, 32, 64, 'bf16').
        gradient_clip_val: Gradient clipping value.
        accumulate_grad_batches: Gradient accumulation steps.
        check_val_every_n_epoch: Validate every N epochs.
        log_every_n_steps: Log every N steps.
        enable_progress_bar: Whether to show progress bar.
        enable_model_summary: Whether to print model summary.
        limit_train_batches: Limit training batches for debugging.
        limit_val_batches: Limit validation batches for debugging.
        limit_test_batches: Limit test batches for debugging.
        default_root_dir: Default root directory for logs.
        save_dir: Directory to save checkpoints and logs.
        experiment_name: Name of the experiment.
        early_stopping_patience: Patience for early stopping (0 to disable).
        save_top_k: Number of best checkpoints to save.
        monitor_metric: Metric to monitor for checkpointing.
        monitor_mode: Mode for monitoring ('min' or 'max').
    """

    # Training
    max_epochs: int = 100
    min_epochs: int = 1
    accelerator: str = "auto"
    devices: Union[int, List[int], str] = "auto"
    precision: Union[int, str] = 32
    gradient_clip_val: Optional[float] = None
    accumulate_grad_batches: int = 1

    # Validation and logging
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 50
    enable_progress_bar: bool = False
    enable_model_summary: bool = True
    limit_train_batches: Optional[Union[int, float]] = None
    limit_val_batches: Optional[Union[int, float]] = None
    limit_test_batches: Optional[Union[int, float]] = None

    # Paths
    default_root_dir: Optional[str] = None
    save_dir: str = "./logs"
    experiment_name: str = "tsflib_experiment"

    # Callbacks
    early_stopping_patience: int = 10
    save_top_k: int = 1
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"

    # Profiling
    profiling: Optional[List[str]] = None  # e.g. ['all'] or ['epoch_time', 'num_params']

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
        }


class TSTrainer:
    """High-level trainer for time series forecasting models.

    This class wraps PyTorch Lightning Trainer with sensible defaults
    for time series forecasting tasks.

    Args:
        config: Trainer configuration.
        callbacks: Additional callbacks.
        logger: Logger instance(s).

    Example:
        >>> config = TSTrainerConfig(
        ...     max_epochs=100,
        ...     accelerator='gpu',
        ...     devices=1,
        ...     early_stopping_patience=10
        ... )
        >>> trainer = TSTrainer(config)
        >>> trainer.fit(model, datamodule=datamodule)
        >>> trainer.test(model, datamodule=datamodule)
    """

    def __init__(
        self,
        config: Optional[TSTrainerConfig] = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[Union[Logger, List[Logger]]] = None,
    ):
        self.config = config or TSTrainerConfig()
        self.callbacks = callbacks or []
        self._logger = logger
        self._trainer: Optional[Trainer] = None
        self._setup_trainer()

    def _setup_trainer(self) -> None:
        """Setup the PyTorch Lightning Trainer."""
        # Create callbacks
        all_callbacks = self._create_callbacks()
        all_callbacks.extend(self.callbacks)

        # In notebook, disable Rich (causes recursion error) and use TQDM
        notebook = _is_notebook()
        enable_summary = self.config.enable_model_summary

        if notebook:
            # Use TQDM instead of Rich to avoid recursion in Jupyter
            all_callbacks.append(TQDMProgressBar(refresh_rate=self.config.log_every_n_steps))
            # Disable Rich model summary (also uses Rich)
            enable_summary = False
            # Disable the default progress bar (which is Rich-based)
            os.environ["RICH_ENABLED"] = "0"

        # Create logger
        pl_logger = self._create_logger()

        # Create trainer
        self._trainer = Trainer(
            max_epochs=self.config.max_epochs,
            min_epochs=self.config.min_epochs,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            precision=self.config.precision,
            gradient_clip_val=self.config.gradient_clip_val,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            check_val_every_n_epoch=self.config.check_val_every_n_epoch,
            log_every_n_steps=self.config.log_every_n_steps,
            enable_progress_bar=self.config.enable_progress_bar,
            enable_model_summary=enable_summary,
            limit_train_batches=self.config.limit_train_batches,
            limit_val_batches=self.config.limit_val_batches,
            limit_test_batches=self.config.limit_test_batches,
            default_root_dir=self.config.default_root_dir,
            callbacks=all_callbacks,
            logger=pl_logger,
            num_sanity_val_steps=0,
        )

    def _create_callbacks(self) -> List[Callback]:
        """Create default callbacks."""
        callbacks = []

        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            save_top_k=self.config.save_top_k,
            filename="epoch{epoch:03d}-" + f"{{{self.config.monitor_metric}:.3f}}",
            auto_insert_metric_name=False,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        if self.config.early_stopping_patience > 0:
            early_stop_callback = EarlyStopping(
                monitor=self.config.monitor_metric,
                mode=self.config.monitor_mode,
                patience=self.config.early_stopping_patience,
                verbose=True,
            )
            callbacks.append(early_stop_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # Training timer
        timer = TrainingTimer()
        callbacks.append(timer)

        # Metrics callback
        metrics_callback = MetricsCallback()
        callbacks.append(metrics_callback)

        # Epoch metrics printer (replaces progress bar)
        epoch_printer = EpochMetricsPrinter()
        callbacks.append(epoch_printer)

        # Profiling callback (only when user opts in)
        if self.config.profiling:
            profiling_callback = ProfilingCallback(enabled=self.config.profiling)
            callbacks.append(profiling_callback)

        return callbacks

    def _create_logger(self) -> List[Logger]:
        """Create loggers.

        Uses a single TensorBoardLogger as the primary logger so that only
        one version directory is created per run.  A CSVLogger is attached
        that reuses the *same* version directory (by pointing its
        ``save_dir`` to the TensorBoard log_dir) so that ``metrics.csv`` and
        ``hparams.yaml`` land alongside the TensorBoard events file instead
        of spawning a second version directory.
        """
        if self._logger is not None:
            if isinstance(self._logger, list):
                return self._logger
            return [self._logger]

        # Primary logger – determines the version directory
        tb_logger = TensorBoardLogger(
            save_dir=self.config.save_dir,
            name=self.config.experiment_name,
        )

        # CSV logger reuses the same log_dir so no extra version dir is created
        csv_logger = CSVLogger(
            save_dir=tb_logger.log_dir,
            name="",
            version="",
        )

        return [tb_logger, csv_logger]

    def fit(
        self,
        model: L.LightningModule,
        datamodule: Optional[L.LightningDataModule] = None,
        train_dataloaders: Optional[Any] = None,
        val_dataloaders: Optional[Any] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        """Train the model.

        Args:
            model: Model to train.
            datamodule: Data module with train/val data.
            train_dataloaders: Training data loader (if not using datamodule).
            val_dataloaders: Validation data loader (if not using datamodule).
            ckpt_path: Path to checkpoint to resume from.
        """
        if self._trainer is None:
            raise RuntimeError("Trainer not initialized")

        self._trainer.fit(
            model=model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

    def validate(
        self,
        model: Optional[L.LightningModule] = None,
        datamodule: Optional[L.LightningDataModule] = None,
        dataloaders: Optional[Any] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Validate the model.

        Args:
            model: Model to validate.
            datamodule: Data module with validation data.
            dataloaders: Validation data loader.
            ckpt_path: Path to checkpoint to load.
            verbose: Whether to print progress.

        Returns:
            Validation results.
        """
        if self._trainer is None:
            raise RuntimeError("Trainer not initialized")

        return self._trainer.validate(
            model=model,
            dataloaders=dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            verbose=verbose,
        )

    def test(
        self,
        model: Optional[L.LightningModule] = None,
        datamodule: Optional[L.LightningDataModule] = None,
        dataloaders: Optional[Any] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Test the model.

        Args:
            model: Model to test.
            datamodule: Data module with test data.
            dataloaders: Test data loader.
            ckpt_path: Path to checkpoint to load.
            verbose: Whether to print progress.

        Returns:
            Test results.
        """
        if self._trainer is None:
            raise RuntimeError("Trainer not initialized")

        return self._trainer.test(
            model=model,
            dataloaders=dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            verbose=verbose,
            weights_only=False,
        )

    def predict(
        self,
        model: Optional[L.LightningModule] = None,
        datamodule: Optional[L.LightningDataModule] = None,
        dataloaders: Optional[Any] = None,
        ckpt_path: Optional[str] = None,
        return_predictions: bool = True,
    ) -> Any:
        """Generate predictions.

        Args:
            model: Model to use for prediction.
            datamodule: Data module with prediction data.
            dataloaders: Prediction data loader.
            ckpt_path: Path to checkpoint to load.
            return_predictions: Whether to return predictions.

        Returns:
            Predictions if return_predictions is True.
        """
        if self._trainer is None:
            raise RuntimeError("Trainer not initialized")

        return self._trainer.predict(
            model=model,
            dataloaders=dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            return_predictions=return_predictions,
        )

    @property
    def checkpoint_callback(self) -> Optional[ModelCheckpoint]:
        """Get the checkpoint callback."""
        if self._trainer is None:
            return None
        return self._trainer.checkpoint_callback

    @property
    def current_epoch(self) -> int:
        """Get current epoch."""
        if self._trainer is None:
            return 0
        return self._trainer.current_epoch

    @property
    def global_step(self) -> int:
        """Get global step."""
        if self._trainer is None:
            return 0
        return self._trainer.global_step

    @property
    def logger(self) -> Optional[Logger]:
        """Get the logger."""
        if self._trainer is None or not self._trainer.loggers:
            return None
        return self._trainer.loggers[0]

    @property
    def log_dir(self) -> Optional[str]:
        """Get the log directory."""
        logger = self.logger
        if logger is None:
            return None
        return getattr(logger, "log_dir", None)

    def save_checkpoint(
        self,
        filepath: str,
        model: Optional[L.LightningModule] = None,
    ) -> None:
        """Save a checkpoint manually.

        Args:
            filepath: Path to save the checkpoint.
            model: Model to save (uses current model if None).
        """
        if self._trainer is None:
            raise RuntimeError("Trainer not initialized")

        self._trainer.save_checkpoint(filepath, weights_only=False)


class TSTrainerBuilder:
    """Builder for creating TSTrainer with preset configurations.

    Example:
        >>> builder = TSTrainerBuilder()
        >>> trainer = builder.quick()  # Quick training
        >>> trainer = builder.gpu()    # GPU training
        >>> trainer = builder.debug()  # Debug mode
    """

    @staticmethod
    def quick(
        max_epochs: int = 10,
        save_dir: str = "./logs",
        experiment_name: str = "quick_run",
    ) -> TSTrainer:
        """Create a trainer for quick experiments."""
        config = TSTrainerConfig(
            max_epochs=max_epochs,
            save_dir=save_dir,
            experiment_name=experiment_name,
            enable_progress_bar=True,
            early_stopping_patience=0,  # Disable early stopping
            save_top_k=1,
        )
        return TSTrainer(config)

    @staticmethod
    def gpu(
        max_epochs: int = 100,
        devices: int = 1,
        precision: Union[int, str] = 32,
        save_dir: str = "./logs",
        experiment_name: str = "gpu_run",
    ) -> TSTrainer:
        """Create a trainer for GPU training."""
        config = TSTrainerConfig(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=devices,
            precision=precision,
            save_dir=save_dir,
            experiment_name=experiment_name,
            early_stopping_patience=10,
        )
        return TSTrainer(config)

    @staticmethod
    def debug(
        max_epochs: int = 1,
        fast_dev_run: bool = True,
    ) -> TSTrainer:
        """Create a trainer for debugging."""
        config = TSTrainerConfig(
            max_epochs=max_epochs,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=1,
        )
        trainer = TSTrainer(config)
        if fast_dev_run:
            # Override trainer with fast_dev_run
            trainer._trainer = Trainer(
                fast_dev_run=True,
                enable_progress_bar=True,
            )
        return trainer

    @staticmethod
    def distributed(
        max_epochs: int = 100,
        devices: int = 2,
        strategy: str = "ddp",
        save_dir: str = "./logs",
        experiment_name: str = "distributed_run",
    ) -> TSTrainer:
        """Create a trainer for distributed training."""
        config = TSTrainerConfig(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=devices,
            save_dir=save_dir,
            experiment_name=experiment_name,
        )
        trainer = TSTrainer(config)
        # Override with distributed strategy
        trainer._trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=devices,
            strategy=strategy,
            default_root_dir=save_dir,
        )
        return trainer


__all__ = [
    "TSTrainer",
    "TSTrainerConfig",
    "TSTrainerBuilder",
]
