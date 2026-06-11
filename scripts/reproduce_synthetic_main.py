"""
Reproduce AOSNet/TQNet/CycleNet controlled synthetic results (Table 5).

Covers: Syn-S (stationary), Syn-A (magnitude), Syn-P (alignment), Syn-C (combined).
H=96, lookback=96. TQNet/CycleNet evaluated with P in {12, 24, 36, 48}.

Usage:
    python scripts/reproduce_synthetic_main.py --datasets all --models all
    python scripts/reproduce_synthetic_main.py --datasets aos_stationary --models AOSNet
    python scripts/reproduce_synthetic_main.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tsflib import Preset

MODELS = ["AOSNet", "TQNet", "CycleNet"]
DATASETS = ["aos_stationary", "aos_amplitude", "aos_phase", "aos_mixed"]
PERIODS = [12, 24, 36, 48]

BASE_CONFIG = {
    "d_model": 256, "dropout": 0.1, "hidden_channels": 8, "hidden_kernel": 5,
    "attention_heads": 4, "attention_dropout": 0.1, "embedding_dropout": 0.1,
    "aux_weight": 0.1, "lr": 2e-3, "batch_size": 64, "epochs": 40,
}


def run_one(model: str, dataset: str, period: int, gpu: int = 0, dry_run: bool = False):
    cfg = BASE_CONFIG
    model_overrides = {
        "d_model": cfg["d_model"], "dropout": cfg["dropout"],
        "hidden_channels": cfg["hidden_channels"], "hidden_kernel": cfg["hidden_kernel"],
        "attention_heads": cfg["attention_heads"],
        "attention_dropout": cfg["attention_dropout"],
        "embedding_dropout": cfg["embedding_dropout"],
        "aux_weight": cfg["aux_weight"],
        "env_weight": 0, "phase_weight": 0, "if_weight": 0,
    }
    if model in ("TQNet", "CycleNet"):
        model_overrides["cycle"] = period

    preset_kwargs = dict(
        lookback=96, horizon=96, epochs=cfg["epochs"],
        batch_size=cfg["batch_size"], lr=cfg["lr"], gpu=gpu,
        category="common",
        model_overrides=model_overrides,
        overrides={"train": {"early_stopping_patience": 8},
                   "trainer": {"monitor_metric": "val_loss", "monitor_mode": "min"}},
    )

    period_label = f"_P{period}" if model != "AOSNet" else ""
    tag = f"{model}_{dataset}_H96{period_label}"
    print(f"\n{'='*60}")
    print(f"  {model} | {dataset} | H=96{period_label}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] {json.dumps(preset_kwargs, indent=2, default=str)}")
        return

    save_dir = f"reproduce_results/table5/{tag}/{tag}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    preset_kwargs["save_dir"] = save_dir
    preset_kwargs["experiment_name"] = tag

    preset = Preset(model, dataset, **preset_kwargs)
    preset.run()


def main():
    parser = argparse.ArgumentParser(description="Reproduce controlled synthetic results (Table 5)")
    parser.add_argument("--datasets", nargs="+", default=["aos_stationary"],
                        help="Dataset names or 'all'")
    parser.add_argument("--models", nargs="+", default=["AOSNet"],
                        help="Model names or 'all'")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if "all" in args.datasets else args.datasets
    models = MODELS if "all" in args.models else args.models

    for model in models:
        for ds in datasets:
            if model == "AOSNet":
                run_one(model, ds, period=0, gpu=args.gpu, dry_run=args.dry_run)
            else:
                for p in PERIODS:
                    run_one(model, ds, period=p, gpu=args.gpu, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
