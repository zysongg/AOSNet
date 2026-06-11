"""
Common Utility Functions for TSFLib

This module provides general utility functions used throughout the library.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    This function sets the random seed for Python's random module,
    NumPy, and PyTorch (including CUDA).

    Args:
        seed: Random seed value.

    Example:
        >>> set_seed(42)
        >>> # Now all random operations are reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Get the device to use for computations.

    Args:
        device: Preferred device (None for auto-selection).

    Returns:
        torch.device object.

    Example:
        >>> device = get_device()  # Auto-select GPU if available
        >>> device = get_device("cpu")  # Force CPU
    """
    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Number of parameters.

    Example:
        >>> model = MyModel()
        >>> print(f"Model has {count_parameters(model):,} parameters")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size(model: torch.nn.Module, unit: str = "MB") -> float:
    """Get the memory size of a model.

    Args:
        model: PyTorch model.
        unit: Unit for size ('B', 'KB', 'MB', 'GB').

    Returns:
        Model size in the specified unit.

    Example:
        >>> size_mb = get_model_size(model, unit='MB')
        >>> print(f"Model size: {size_mb:.2f} MB")
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size

    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
    }

    return total_size / units.get(unit, units["MB"])


def save_config(config: Dict[str, Any], path: Union[str, Path], format: str = "yaml") -> None:
    """Save a configuration dictionary to file.

    Args:
        config: Configuration dictionary.
        path: Path to save the file.
        format: File format ('yaml' or 'json').

    Example:
        >>> config = {'lr': 0.001, 'batch_size': 32}
        >>> save_config(config, 'config.yaml')
        >>> save_config(config, 'config.json', format='json')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "yaml":
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    elif format == "json":
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'.")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a configuration from file.

    Args:
        path: Path to the configuration file.

    Returns:
        Configuration dictionary.

    Example:
        >>> config = load_config('config.yaml')
        >>> print(config['lr'])
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists.

    Args:
        path: Directory path.

    Returns:
        Path object.

    Example:
        >>> log_dir = ensure_dir('./logs/experiment_1')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.

    Example:
        >>> print(format_time(3661))  # '1h 1m 1s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_number(num: float, precision: int = 2) -> str:
    """Format a number with appropriate suffix.

    Args:
        num: Number to format.
        precision: Decimal precision.

    Returns:
        Formatted string.

    Example:
        >>> print(format_number(1500))  # '1.50K'
        >>> print(format_number(1500000))  # '1.50M'
    """
    suffixes = ["", "K", "M", "B", "T"]
    suffix_idx = 0

    while abs(num) >= 1000 and suffix_idx < len(suffixes) - 1:
        num /= 1000
        suffix_idx += 1

    return f"{num:.{precision}f}{suffixes[suffix_idx]}"


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage.

    Returns:
        Dictionary with memory usage in MB.

    Example:
        >>> usage = get_memory_usage()
        >>> print(f"Allocated: {usage['allocated']:.2f} MB")
    """
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024 ** 2,
            "reserved": torch.cuda.memory_reserved() / 1024 ** 2,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024 ** 2,
        }
    return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}


def print_model_summary(model: torch.nn.Module, input_size: Optional[tuple] = None) -> None:
    """Print a summary of the model.

    Args:
        model: PyTorch model.
        input_size: Input size tuple (batch, features, seq_len).

    Example:
        >>> model = MyModel()
        >>> print_model_summary(model, input_size=(32, 7, 96))
    """
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)

    # Model architecture
    print("\nArchitecture:")
    print(model)

    # Parameter count
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print("\nParameters:")
    print(f"  Total:      {format_number(total_params, 0)}")
    print(f"  Trainable:  {format_number(trainable_params, 0)}")
    print(f"  Non-trainable: {format_number(total_params - trainable_params, 0)}")

    # Model size
    size_mb = get_model_size(model, unit="MB")
    print(f"\nModel Size: {size_mb:.2f} MB")

    # Try to get input/output shapes
    if input_size is not None:
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(*input_size)
                output = model(dummy_input)
                print(f"\nInput Shape:  {tuple(dummy_input.shape)}")
                print(f"Output Shape: {tuple(output.shape)}")
        except Exception as e:
            print(f"\nCould not infer shapes: {e}")

    print("=" * 80)


def chunk_list(lst: list, chunk_size: int):
    """Split a list into chunks.

    Args:
        lst: List to split.
        chunk_size: Size of each chunk.

    Yields:
        Chunks of the list.

    Example:
        >>> for chunk in chunk_list([1, 2, 3, 4, 5], 2):
        ...     print(chunk)
        [1, 2]
        [3, 4]
        [5]
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average.

    Args:
        data: Input data.
        window: Window size.

    Returns:
        Smoothed data.

    Example:
        >>> smoothed = moving_average(data, window=7)
    """
    if window <= 1:
        return data

    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


__all__ = [
    "set_seed",
    "get_device",
    "count_parameters",
    "get_model_size",
    "save_config",
    "load_config",
    "ensure_dir",
    "format_time",
    "format_number",
    "get_memory_usage",
    "print_model_summary",
    "chunk_list",
    "moving_average",
]
