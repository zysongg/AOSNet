"""
TSFLib Model Configurations

Ready-to-use presets: just specify model name and dataset name.

Example:
    >>> from tsflib.confs import Preset, train
    >>> preset = Preset("DLinear", "etth1")
    >>> preset.run()

    >>> train("DLinear", "etth1")
"""

from tsflib.confs.presets import (
    Preset,
    train,
    list_models,
    list_datasets,
)

__all__ = [
    "Preset",
    "train",
    "list_models",
    "list_datasets",
]
