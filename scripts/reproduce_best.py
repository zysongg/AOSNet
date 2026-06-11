"""
AOSNet Optimal Reproduction Script
Best configurations from reproduction experiments.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsflib import Preset

CONFIGS = {
    "etth1": {"lr": 0.0005, "batch_size": 512, "d_model": 512, "dropout": 0.1, "attn_drop": 0.1, "emb_drop": 0.1, "aux": 0.1, "epochs": 60, "loss_type": "mix_freq_mse", "alpha": 1.0, "monitor": "val_loss"},
    "etth2": {"lr": 0.003, "batch_size": 16, "d_model": 512, "dropout": 0.1, "attn_drop": 0.1, "emb_drop": 0.1, "aux": 0.1, "epochs": 60, "loss_type": "mix_freq_mse", "alpha": 1.0, "monitor": "val_loss"},
    "ettm1": {"lr": 0.003, "batch_size": 512, "d_model": 512, "dropout": 0.1, "attn_drop": 0.1, "emb_drop": 0.1, "aux": 0.1, "epochs": 60, "loss_type": "mix_freq_mse", "alpha": 1.0, "monitor": "val_loss"},
    "ettm2": {"lr": 0.003, "batch_size": 256, "d_model": 512, "dropout": 0.1, "attn_drop": 0.1, "emb_drop": 0.1, "aux": 0.1, "epochs": 60, "loss_type": "mix_freq_mse", "alpha": 1.0, "monitor": "val_loss"},
    "ecl": {"lr": 0.003, "batch_size": 32, "d_model": 512, "dropout": 0.1, "attn_drop": 0.1, "emb_drop": 0.1, "aux": 0, "epochs": 50, "loss_type": "mix_freq_mse", "alpha": 1.0, "monitor": "val_loss"},
    "weather": {"lr": 0.003, "batch_size": 32, "d_model": 512, "dropout": 0.1, "attn_drop": 0.1, "emb_drop": 0.1, "aux": 0, "epochs": 60, "loss_type": "mix_freq_mse", "alpha": 1.0, "monitor": "val_loss"},
    "traffic": {"lr": 0.001, "batch_size": 256, "d_model": 512, "dropout": 0.1, "attn_drop": 0.1, "emb_drop": 0.1, "aux": 0.1, "epochs": 50, "loss_type": "mix_freq_mae", "alpha": 1.0, "monitor": "val_loss"},
    "solar": {"lr": 0.001, "batch_size": 128, "d_model": 512, "dropout": 0.2, "attn_drop": 0.2, "emb_drop": 0.2, "aux": 0, "epochs": 50, "loss_type": "mix_freq_mse", "alpha": 1.0, "monitor": "val_mse"},
}

def reproduce(dataset, horizon=96, gpu=0, dry_run=False):
    import json
    c = CONFIGS[dataset]
    mo = {
        "d_model": c["d_model"], "dropout": c["dropout"],
        "hidden_channels": 8, "hidden_kernel": 5,
        "attention_heads": 4,
        "attention_dropout": c["attn_drop"], "embedding_dropout": c["emb_drop"],
        "aux_weight": c["aux"], "env_weight": 0, "phase_weight": 0, "if_weight": 0,
    }
    ov = {
        "loss": {"loss_type": c["loss_type"], "alpha": c["alpha"]},
        "train": {"early_stopping_patience": 5},
        "trainer": {"monitor_metric": c["monitor"], "monitor_mode": "min"},
    }
    kwargs = dict(lookback=96, horizon=horizon, epochs=c["epochs"],
                  batch_size=c["batch_size"], lr=c["lr"], gpu=gpu,
                  model_overrides=mo, overrides=ov)
    if dry_run:
        print(json.dumps(kwargs, indent=2, default=str))
        return
    preset = Preset("AOSNet", dataset, **kwargs)
    preset.run()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(CONFIGS.keys()))
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for ds in args.datasets:
        print(f"\n{'='*50}\n  AOSNet | {ds} | H={args.horizon}\n{'='*50}")
        reproduce(ds, args.horizon, args.gpu, args.dry_run)
