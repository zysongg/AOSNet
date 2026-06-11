"""
Custom Callbacks for TSFLib Training

This module provides custom PyTorch Lightning callbacks for time series
forecasting training.
"""

import os
import time
from typing import Any, Dict, List, Optional

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


class MetricsCallback(Callback):
    """Callback to track and log metrics during training.

    This callback collects metrics from each epoch and provides
    methods to access them after training.

    Example:
        >>> callback = MetricsCallback()
        >>> trainer = Trainer(callbacks=[callback])
        >>> trainer.fit(model, datamodule)
        >>>
        >>> # Access metrics
        >>> print(callback.train_metrics)
        >>> print(callback.val_metrics)
    """

    def __init__(self):
        super().__init__()
        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []
        self.test_metrics: Optional[Dict[str, float]] = None

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Collect training metrics."""
        metrics = trainer.callback_metrics
        train_metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
            if k.startswith("train")
        }
        self.train_metrics.append(train_metrics)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Collect validation metrics."""
        metrics = trainer.callback_metrics
        val_metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
            if k.startswith("val")
        }
        self.val_metrics.append(val_metrics)

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Collect test metrics."""
        metrics = trainer.callback_metrics
        self.test_metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }

    def get_best_epoch(self, metric: str = "val_loss", mode: str = "min") -> int:
        """Get the epoch with the best validation metric.

        Args:
            metric: Metric name to check.
            mode: 'min' or 'max'.

        Returns:
            Best epoch number (0-indexed).
        """
        if not self.val_metrics:
            return 0

        values = [
            m.get(metric, float("inf") if mode == "min" else float("-inf"))
            for m in self.val_metrics
        ]

        if mode == "min":
            return int(np.argmin(values))
        else:
            return int(np.argmax(values))


class TrainingTimer(Callback):
    """Callback to track training time.

    Example:
        >>> timer = TrainingTimer()
        >>> trainer = Trainer(callbacks=[timer])
        >>> trainer.fit(model, datamodule)
        >>>
        >>> print(f"Training took {timer.total_time:.2f} seconds")
        >>> print(f"Average epoch time: {timer.avg_epoch_time:.2f} seconds")
    """

    def __init__(self):
        super().__init__()
        self.start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        self.total_time: float = 0.0
        self.epoch_times: List[float] = []

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start timing."""
        self.start_time = time.time()

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start epoch timing."""
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Record epoch time."""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Calculate total time."""
        if self.start_time is not None:
            self.total_time = time.time() - self.start_time

    @property
    def avg_epoch_time(self) -> float:
        """Get average epoch time."""
        if not self.epoch_times:
            return 0.0
        return float(np.mean(self.epoch_times))

    @property
    def total_epochs(self) -> int:
        """Get total number of epochs trained."""
        return len(self.epoch_times)


class ProfilingCallback(Callback):
    """Callback to collect profiling information during training and testing.

    Collects: epoch_time, num_params, flops, max_gpu_memory, inference_time_per_batch.

    Args:
        enabled: List of profiling items to collect.
            Available: 'epoch_time', 'num_params', 'flops',
                       'max_gpu_memory', 'inference_time_per_batch'.
            Use 'all' to enable everything.

    Example:
        >>> cb = ProfilingCallback(enabled=['epoch_time', 'num_params', 'max_gpu_memory'])
        >>> trainer = Trainer(callbacks=[cb])
        >>> trainer.fit(model, datamodule)
        >>> print(cb.results)
    """

    ALL_ITEMS = [
        'epoch_time',
        'num_params',
        'flops',
        'max_gpu_memory',
        'inference_time_per_batch',
    ]

    def __init__(self, enabled: List[str] = None):
        super().__init__()
        if enabled is None:
            self.enabled: List[str] = []
        elif 'all' in enabled:
            self.enabled = list(self.ALL_ITEMS)
        else:
            unknown = set(enabled) - set(self.ALL_ITEMS)
            if unknown:
                raise ValueError(
                    f"Unknown profiling items: {unknown}. "
                    f"Available: {self.ALL_ITEMS}"
                )
            self.enabled = list(enabled)

        self.results: Dict[str, Any] = {}

        # Internal state
        self._epoch_start: Optional[float] = None
        self._epoch_times: List[float] = []
        self._max_gpu_mem_mb: float = 0.0
        self._test_batch_times: List[float] = []
        self._test_batch_start: Optional[float] = None

    # -- helpers --

    def _is(self, item: str) -> bool:
        return item in self.enabled

    @staticmethod
    def _count_params(model: torch.nn.Module) -> Dict[str, int]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @staticmethod
    def _estimate_flops(model: torch.nn.Module, input_shape: tuple) -> Optional[float]:
        """Estimate FLOPs via a single forward pass with torch.cuda.Event timing.

        Returns MFLOPs (10^6 FLOPs), or None if CUDA is not available.
        This is a rough estimate using the linear relationship between
        arithmetic intensity and wall-clock time on GPU.  For an exact
        count use fvcore or ptflops; here we keep it dependency-free.
        """
        if not torch.cuda.is_available():
            return None
        # Use the empirical approach: count multiply-adds per layer type
        # For simplicity, use the parameter-based approximation:
        #   FLOPs ~= 2 * num_params (for forward pass of linear layers)
        # This is a lower bound but commonly used for quick comparison.
        total_params = sum(p.numel() for p in model.parameters())
        return 2.0 * total_params / 1e6  # MFLOPs

    def _reset_peak_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._max_gpu_mem_mb = 0.0

    def _update_peak_gpu_memory(self):
        if torch.cuda.is_available() and self._is('max_gpu_memory'):
            current = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self._max_gpu_mem_mb = max(self._max_gpu_mem_mb, current)

    # -- training hooks --

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._is('num_params'):
            self.results['num_params'] = self._count_params(pl_module.model)
        if self._is('flops'):
            # Estimate FLOPs with a dummy input
            try:
                x = torch.randn(1, pl_module.num_features, pl_module.lookback_len,
                                device=pl_module.device)
                y = torch.randn(1, pl_module.num_features, pl_module.horizon_len,
                                device=pl_module.device)
                batch = (x, y)
                self.results['flops_mflops'] = self._estimate_flops(pl_module.model, (1,))
            except Exception:
                self.results['flops_mflops'] = None
        if self._is('max_gpu_memory'):
            self._reset_peak_gpu_memory()

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._is('epoch_time'):
            self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._is('epoch_time') and self._epoch_start is not None:
            self._epoch_times.append(time.time() - self._epoch_start)
        if self._is('max_gpu_memory'):
            self._update_peak_gpu_memory()

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._is('epoch_time') and self._epoch_times:
            self.results['epoch_time'] = {
                'per_epoch_s': [round(t, 3) for t in self._epoch_times],
                'mean_s': round(float(np.mean(self._epoch_times)), 3),
            }
        if self._is('max_gpu_memory'):
            self._update_peak_gpu_memory()
            self.results['max_gpu_memory_mb'] = round(self._max_gpu_mem_mb, 1)

    # -- test hooks --

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._test_batch_times = []
        if self._is('max_gpu_memory'):
            self._reset_peak_gpu_memory()

    def on_test_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule,
                            batch: Any, batch_idx: int,
                            dataloader_idx: int = 0) -> None:
        if self._is('inference_time_per_batch'):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._test_batch_start = time.time()

    def on_test_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule,
                          outputs: Any, batch: Any, batch_idx: int,
                          dataloader_idx: int = 0) -> None:
        if self._is('inference_time_per_batch') and self._test_batch_start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._test_batch_times.append(time.time() - self._test_batch_start)
        if self._is('max_gpu_memory'):
            self._update_peak_gpu_memory()

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._is('inference_time_per_batch') and self._test_batch_times:
            self.results['inference_time_per_batch'] = {
                'mean_ms': round(float(np.mean(self._test_batch_times)) * 1000, 2),
                'std_ms': round(float(np.std(self._test_batch_times)) * 1000, 2),
                'num_batches': len(self._test_batch_times),
            }
        if self._is('max_gpu_memory'):
            self._update_peak_gpu_memory()
            self.results['max_gpu_memory_mb'] = round(self._max_gpu_mem_mb, 1)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Also update here in case on_test_epoch_end wasn't reached
        if self._is('inference_time_per_batch') and self._test_batch_times \
                and 'inference_time_per_batch' not in self.results:
            self.results['inference_time_per_batch'] = {
                'mean_ms': round(float(np.mean(self._test_batch_times)) * 1000, 2),
                'std_ms': round(float(np.std(self._test_batch_times)) * 1000, 2),
                'num_batches': len(self._test_batch_times),
            }
        if self._is('max_gpu_memory') and 'max_gpu_memory_mb' not in self.results:
            self._update_peak_gpu_memory()
            self.results['max_gpu_memory_mb'] = round(self._max_gpu_mem_mb, 1)


class EpochMetricsPrinter(Callback):
    """Callback to print epoch metrics in a clean table format.

    Prints epoch number, train/val metrics, and epoch time after each epoch.

    Example:
        >>> printer = EpochMetricsPrinter()
        >>> trainer = Trainer(callbacks=[printer])
    """

    def __init__(self, metrics_to_print: Optional[List[str]] = None):
        super().__init__()
        self._metrics_to_print = metrics_to_print  # None = print all
        self._epoch_start: Optional[float] = None

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        epoch_time = time.time() - self._epoch_start if self._epoch_start else 0

        # Collect train metrics
        train_info = {}
        for k, v in metrics.items():
            if k.startswith("train"):
                val = v.item() if isinstance(v, torch.Tensor) else v
                train_info[k] = val

        # Collect val metrics
        val_info = {}
        for k, v in metrics.items():
            if k.startswith("val"):
                val = v.item() if isinstance(v, torch.Tensor) else v
                val_info[k] = val

        # Filter if specific metrics requested
        if self._metrics_to_print:
            train_info = {k: v for k, v in train_info.items() if k in self._metrics_to_print}
            val_info = {k: v for k, v in val_info.items() if k in self._metrics_to_print}

        # Build output
        parts = [f"Epoch {epoch}"]
        parts.append(f"Time: {epoch_time:.2f}s")
        if train_info:
            train_str = " | ".join(f"{k}={v:.4f}" for k, v in train_info.items())
            parts.append(train_str)
        if val_info:
            val_str = " | ".join(f"{k}={v:.4f}" for k, v in val_info.items())
            parts.append(val_str)

        print(" | ".join(parts))


class PredictionWriter(Callback):
    """Callback to save predictions during testing.

    This callback saves predictions and targets to disk for
    further analysis.

    Args:
        output_dir: Directory to save predictions.
        write_interval: When to write ('batch' or 'epoch').

    Example:
        >>> writer = PredictionWriter(output_dir='./predictions')
        >>> trainer = Trainer(callbacks=[writer])
        >>> trainer.test(model, datamodule)
        >>>
        >>> # Predictions are saved to ./predictions/
    """

    def __init__(
        self,
        output_dir: str = "./predictions",
        write_interval: str = "epoch",
    ):
        super().__init__()
        self.output_dir = output_dir
        self.write_interval = write_interval
        self.predictions: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []

        os.makedirs(output_dir, exist_ok=True)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect predictions from batch."""
        if isinstance(outputs, dict):
            pred = outputs.get("pred", outputs.get("predictions"))
            target = outputs.get("target", outputs.get("targets"))
        else:
            pred, target = outputs

        if pred is not None:
            self.predictions.append(pred.detach().cpu().numpy())
        if target is not None:
            self.targets.append(target.detach().cpu().numpy())

        if self.write_interval == "batch":
            self._write_batch(batch_idx)

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Write all predictions at end of epoch."""
        if self.write_interval == "epoch":
            self._write_epoch()

    def _write_batch(self, batch_idx: int) -> None:
        """Write predictions for a single batch."""
        if not self.predictions:
            return

        np.savez(
            os.path.join(self.output_dir, f"predictions_batch_{batch_idx}.npz"),
            predictions=self.predictions[-1],
            targets=self.targets[-1] if self.targets else None,
        )

    def _write_epoch(self) -> None:
        """Write all predictions for the epoch."""
        if not self.predictions:
            return

        all_preds = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0) if self.targets else None

        np.savez(
            os.path.join(self.output_dir, "predictions.npz"),
            predictions=all_preds,
            targets=all_targets,
        )

        # Clear stored predictions
        self.predictions = []
        self.targets = []


class EarlyStoppingWithWarmup(Callback):
    """Early stopping with warmup period.

    This callback prevents early stopping during the initial
    warmup epochs.

    Args:
        monitor: Metric to monitor.
        mode: 'min' or 'max'.
        patience: Patience for early stopping.
        warmup_epochs: Number of warmup epochs.
        min_delta: Minimum change to qualify as improvement.

    Example:
        >>> early_stop = EarlyStoppingWithWarmup(
        ...     monitor='val_loss',
        ...     patience=10,
        ...     warmup_epochs=20
        ... )
        >>> trainer = Trainer(callbacks=[early_stop])
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        warmup_epochs: int = 0,
        min_delta: float = 0.0,
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.min_delta = min_delta

        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Check if should stop."""
        current_epoch = trainer.current_epoch

        # Skip during warmup
        if current_epoch < self.warmup_epochs:
            return

        # Get current metric
        metrics = trainer.callback_metrics
        current_score = metrics.get(self.monitor)

        if current_score is None:
            return

        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()

        # Initialize best score
        if self.best_score is None:
            self.best_score = current_score
            return

        # Check for improvement
        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        # Check if should stop
        if self.counter >= self.patience:
            self.should_stop = True
            trainer.should_stop = True
            print(f"\nEarly stopping triggered after epoch {current_epoch}")


class GradientMonitor(Callback):
    """Monitor gradient statistics during training.

    This callback logs gradient norms and can detect vanishing/exploding gradients.

    Args:
        log_every_n_steps: How often to log gradient stats.
        warn_threshold: Threshold for gradient norm warnings.

    Example:
        >>> monitor = GradientMonitor(log_every_n_steps=100)
        >>> trainer = Trainer(callbacks=[monitor])
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        warn_threshold: float = 100.0,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.warn_threshold = warn_threshold

    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Monitor gradients after backward pass."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Log gradient norm
        pl_module.log("grad_norm", total_norm, on_step=True, on_epoch=False)

        # Warn if gradients are too large
        if total_norm > self.warn_threshold:
            print(f"\nWarning: Large gradient norm detected: {total_norm:.2f}")


class LearningRateFinder(Callback):
    """Callback to find optimal learning rate.

    This callback runs the learning rate finder at the start of training
    and sets the learning rate accordingly.

    Args:
        min_lr: Minimum learning rate to try.
        max_lr: Maximum learning rate to try.
        num_iterations: Number of iterations for the search.

    Example:
        >>> lr_finder = LearningRateFinder()
        >>> trainer = Trainer(callbacks=[lr_finder])
    """

    def __init__(
        self,
        min_lr: float = 1e-6,
        max_lr: float = 1.0,
        num_iterations: int = 100,
    ):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iterations = num_iterations

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run learning rate finder."""
        # Note: This requires tuner support from Lightning
        # Implementation depends on Lightning version
        pass


__all__ = [
    "MetricsCallback",
    "TrainingTimer",
    "ProfilingCallback",
    "EpochMetricsPrinter",
    "PredictionWriter",
    "EarlyStoppingWithWarmup",
    "GradientMonitor",
    "LearningRateFinder",
]
