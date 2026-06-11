"""
Reproduce AOSNet/TQNet/CycleNet workload forecasting results.

Covers: IaaS / PaaS workload traces, H=96/192/336/720.

Usage:
    python scripts/reproduce_workload_main.py --datasets iaas paas --models AOSNet TQNet CycleNet
    python scripts/reproduce_workload_main.py --datasets all --models all --horizons all
    python scripts/reproduce_workload_main.py --dry-run
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

WORKLOAD_CONFIGS = {
    "iaas": {
        "d_model": 256, "dropout": 0.1, "hidden_channels": 8, "hidden_kernel": 5,
        "attention_heads": 4, "attention_dropout": 0.1, "embedding_dropout": 0.1,
        "aux_weight": 0, "lr": 1e-3, "batch_size": 128,
        "horizons": [96, 192, 336, 720],
        "category": "workload",
    },
    "paas": {
        "d_model": 256, "dropout": 0.1, "hidden_channels": 8, "hidden_kernel": 5,
        "attention_heads": 4, "attention_dropout": 0.1, "embedding_dropout": 0.1,
        "aux_weight": 0, "lr": 1e-3, "batch_size": 128,
        "horizons": [96, 192, 336, 720],
        "category": "workload",
    },
}

ALL_DATASETS = list(WORKLOAD_CONFIGS.keys())


def run_one(model: str, dataset: str, horizon: int, gpu: int = 0, epochs: int = 50, dry_run: bool = False):
    cfg = WORKLOAD_CONFIGS[dataset]
    model_overrides = {
        "d_model": cfg["d_model"], "dropout": cfg["dropout"],
        "hidden_channels": cfg["hidden_channels"], "hidden_kernel": cfg["hidden_kernel"],
        "attention_heads": cfg["attention_heads"],
        "attention_dropout": cfg["attention_dropout"],
        "embedding_dropout": cfg["embedding_dropout"],
        "aux_weight": cfg.get("aux_weight", 0),
        "env_weight": 0, "phase_weight": 0, "if_weight": 0,
    }
    if model in ("TQNet", "CycleNet"):
        model_overrides["cycle"] = 168

    preset_kwargs = dict(
        lookback=96, horizon=horizon, epochs=epochs,
        batch_size=cfg["batch_size"], lr=cfg["lr"], gpu=gpu,
        category=cfg["category"],
        model_overrides=model_overrides,
        overrides={"train": {"early_stopping_patience": 5},
                   "trainer": {"monitor_metric": "val_loss", "monitor_mode": "min"}},
    )

    tag = f"{model}_{dataset}_H{horizon}"
    print(f"\n{'='*60}")
    print(f"  {model} | {dataset} | H={horizon}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] {json.dumps(preset_kwargs, indent=2, default=str)}")
        return

    save_dir = f"reproduce_results/table3/{tag}/{tag}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    preset_kwargs["save_dir"] = save_dir
    preset_kwargs["experiment_name"] = tag

    preset = Preset(model, dataset, **preset_kwargs)
    preset.run()


def main():
    parser = argparse.ArgumentParser(description="Reproduce workload forecasting results")
    parser.add_argument("--datasets", nargs="+", default=["iaas"],
                        help="Dataset names or 'all'")
    parser.add_argument("--models", nargs="+", default=["AOSNet"],
                        help="Model names or 'all'")
    parser.add_argument("--horizons", nargs="+", default=None,
                        help="Horizon lengths or 'all'")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets
    models = MODELS if "all" in args.models else args.models

    for model in models:
        for ds in datasets:
            if ds not in WORKLOAD_CONFIGS:
                print(f"Warning: unknown dataset '{ds}', skipping.")
                continue
            if args.horizons is None or "all" in args.horizons:
                horizons = WORKLOAD_CONFIGS[ds]["horizons"]
            else:
                horizons = [int(h) for h in args.horizons]
            for h in horizons:
                run_one(model, ds, h, gpu=args.gpu, epochs=args.epochs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
