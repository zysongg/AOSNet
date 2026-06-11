"""
Quick start example for AOSNet (AOSNet).

Demonstrates basic usage with the ETT dataset.

Usage:
    cd <project_root>
    python examples/quick_start.py
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tsflib import Preset


def basic_example():
    """Minimal example: train AOSNet on ETT-h1 with horizon 96."""
    print("=== Basic Example: AOSNet on ETT-h1, H=96 ===")
    preset = Preset(
        "AOSNet",
        "etth1",
        horizon=96,
        epochs=5,
        batch_size=32,
        lr=5e-4,
        gpu=0,
        model_overrides={
            "d_model": 128,
            "dropout": 0.1,
            "hidden_channels": 8,
            "hidden_kernel": 5,
            "attention_heads": 4,
            "attention_dropout": 0.1,
            "embedding_dropout": 0.1,
            "env_weight": 0,
            "phase_weight": 0,
            "if_weight": 0,
        },
        overrides={
            "train": {"early_stopping_patience": 3},
        },
    )
    preset.run()


def multi_horizon_example():
    """Run multiple horizons on ETT-h1."""
    print("=== Multi-Horizon Example ===")
    for horizon in [96, 192, 336]:
        print(f"\n--- Horizon {horizon} ---")
        preset = Preset(
            "AOSNet",
            "etth1",
            horizon=horizon,
            epochs=5,
            batch_size=32,
            lr=5e-4,
            gpu=0,
            overrides={
                "train": {"early_stopping_patience": 3},
            },
        )
        preset.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", action="store_true", help="Run multi-horizon example")
    args = parser.parse_args()

    if args.multi:
        multi_horizon_example()
    else:
        basic_example()
