"""
AOSNet (Adaptive Oscillatory Structure Network) — Open-Source Release

Minimal TSFLib-based library for reproducing AOSNet/AOSNet experiments
on long-term and workload forecasting benchmarks.

Example:
    >>> from tsflib import Preset
    >>> preset = Preset("AOSNet", "etth1", horizon=96)
    >>> preset.run()
"""

__version__ = "0.1.0"
__author__ = "AOSNet Contributors"

from tsflib import (
    data,
    layers,
    modules,
    losses,
    metrics,
    trainers,
    utils,
    configs,
    visual,
)

from tsflib.models import (
    ModelBase,
    list_models,
    get_model,
    create_model,
)

from tsflib.data import DataModule, Dataset
from tsflib.trainers import TSTrainer, TSEvaluator

from tsflib.losses import (
    MSELoss,
    MAELoss,
    MixFreqMSELoss,
    MixFreqMAELoss,
)

from tsflib.modules import RevIN

from tsflib.confs import train, Preset
from tsflib.confs.presets import (
    list_models as list_mlp_models,
    list_datasets as list_available_datasets,
)

__all__ = [
    "data", "layers", "modules", "losses", "metrics",
    "trainers", "utils", "configs", "visual",
    "ModelBase", "DataModule", "Dataset",
    "TSTrainer", "TSEvaluator",
    "list_models", "get_model", "create_model",
    "MSELoss", "MAELoss", "MixFreqMSELoss", "MixFreqMAELoss",
    "RevIN",
    "train", "Preset", "list_available_datasets",
]
