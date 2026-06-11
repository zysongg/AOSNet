"""
AOSNet — Adaptive Oscillatory Structure Network convenience package.

Provides a high-level wrapper around AOSNet for quick experimentation.

Example:
    >>> from aosnet import AOSNet, run_experiment
    >>> model = AOSNet(dataset="etth1", horizon=96)
    >>> model.train()

    >>> run_experiment(dataset="etth1", horizons=[96, 192, 336, 720])
"""

from tsflib import Preset

__version__ = "0.1.0"


class AOSNet:
    """High-level interface for AOSNet experiments.

    Args:
        dataset: Dataset name (e.g. 'etth1', 'weather', 'traffic').
        horizon: Forecast horizon length.
        lookback: Input sequence length (default 96).
        lr: Learning rate.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        gpu: GPU device index (None for CPU).
        **kwargs: Additional overrides passed to Preset.
    """

    def __init__(
        self,
        dataset: str = "etth1",
        horizon: int = 96,
        lookback: int = 96,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        gpu: int = 0,
        **kwargs,
    ):
        self.preset = Preset(
            "AOSNet",
            dataset,
            lookback=lookback,
            horizon=horizon,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            gpu=gpu,
            **kwargs,
        )

    def train(self):
        """Run training and evaluation."""
        return self.preset.run()


def run_experiment(
    dataset: str,
    horizons: list = None,
    **kwargs,
):
    """Run AOSNet on multiple horizons for a dataset.

    Args:
        dataset: Dataset name.
        horizons: List of forecast horizons (default: [96, 192, 336, 720]).
        **kwargs: Additional arguments passed to AOSNet.

    Returns:
        List of results per horizon.
    """
    if horizons is None:
        horizons = [96, 192, 336, 720]

    results = []
    for h in horizons:
        model = AOSNet(dataset=dataset, horizon=h, **kwargs)
        result = model.train()
        results.append({"horizon": h, "result": result})
    return results


__all__ = ["AOSNet", "run_experiment"]
