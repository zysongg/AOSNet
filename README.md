# AOSNet — Adaptive Oscillatory-State Alignment for Time Series Forecasting

**Authors:** Zhangyao Song, Chaofeng Qu, Chao Zha, Xiaoyu Zhao, Yinfei Xu, Tao Guo

## Abstract

Long-term time series forecasting benefits from inductive biases that expose recurring temporal structure. Existing periodic forecasting methods typically model recurrence through predefined periods, global spectral components, or fixed learnable templates. However, real-world temporal dynamics are rarely rigidly periodic: around a nominal cycle, oscillatory behavior often exhibits *non-rigid periodicity* (NRP), where cycle magnitude, cycle alignment, and local cycle duration vary over time. Under these conditions, fixed-template periodic modeling can become fundamentally mismatched to the underlying temporal states. We propose AOSNet, a Hilbert-guided forecasting framework that reformulates periodic forecasting from fixed template matching to adaptive oscillatory-state alignment. AOSNet extracts analytic-signal descriptors from both the observed sequence and a learnable global oscillatory prior, then adaptively aligns local states through a descriptor-conditioned gate that selectively preserves reliable observations while softly correcting mismatched regions. The learned prior serves not as a rigid repeated template but as a flexible oscillatory reference interpreted through local state dynamics. Experiments on eight public benchmarks and two cloud workload traces demonstrate leading or highly competitive accuracy with a compact model size and low inference latency, supporting repeated forecasting settings such as capacity planning and autoscaling. Controlled synthetic studies that isolate cycle-magnitude and cycle-alignment variation and combine them with cycle-duration changes show that the advantage of oscillatory-state alignment increases as NRP intensifies.

**Keywords:** Time series forecasting, periodicity, Hilbert transform, analytic signal, non-rigid periodicity.

## Setup

Python >= 3.10, CUDA-capable GPU recommended.

```bash
pip install -r requirements.txt
```

## Data

Download benchmark datasets and place them under `./dataset/`:

| Category | Datasets | Source |
|----------|----------|--------|
| common | ETT (h1/h2/m1/m2), ECL, Solar-Energy, Traffic, Weather | [Autoformer](https://github.com/thuml/Autoformer) / [Monash](https://github.com/rakshitha123/TSForecasting) |
| workload | IaaS, PaaS | Cloud workload traces (10-min interval) |
| synthetic | aos_stationary, aos_amplitude, aos_phase, aos_mixed | Generated via `scripts/reproduce_synthetic_main.py` |

Set `TSDATADIR` environment variable to the data root:
```bash
export TSDATADIR=/path/to/dataset
```

## Quick Start

```python
from tsflib import Preset

preset = Preset("AOSNet", "etth1", horizon=96)
preset.run()
```

## Reproduction Scripts

### Table 2: Long-Term Forecasting (8 datasets, best configs)

```bash
python scripts/reproduce_best.py --datasets all --gpu 0
python scripts/reproduce_best.py --datasets etth1 etth2 --gpu 0
python scripts/reproduce_best.py --dry-run  # preview configs
```

### Table 3: Workload Forecasting (AOSNet / TQNet / CycleNet, H=96-720)

```bash
python scripts/reproduce_workload_main.py --datasets all --models all --horizons all
python scripts/reproduce_workload_main.py --datasets iaas --models AOSNet --horizons 96 192
```

### Table 5: Controlled Synthetic (AOSNet + CycleNet/TQNet with P=12/24/36/48)

```bash
python scripts/reproduce_synthetic_main.py --datasets all --models all
python scripts/reproduce_synthetic_main.py --datasets aos_stationary --models AOSNet
```

## Project Structure

```
AOSNet/
├── aosnet/                  # High-level convenience wrapper
├── tsflib/                  # Core library
│   ├── confs/               # Preset configurations
│   ├── data/                # Data loading and preprocessing
│   ├── layers/              # Neural network layers
│   ├── losses/              # Loss functions (MSE, MAE, MixFreq*)
│   ├── metrics/             # Evaluation metrics
│   ├── models/mlp/          # AOSNet, CycleNet, TQNet
│   ├── modules/             # RevIN, embeddings, patching
│   ├── trainers/            # PyTorch Lightning trainer
│   ├── utils/               # Utilities
│   ├── visual/              # Visualization
│   └── wrappers/            # Lightning module wrappers
├── scripts/
│   ├── reproduce_best.py              # Table 2 best configs
│   ├── reproduce_workload_main.py     # Table 3 workload
│   └── reproduce_synthetic_main.py    # Table 5 synthetic
├── examples/quick_start.py
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@article{song2026aosnet,
  title={Adaptive Oscillatory-State Alignment for Time Series Forecasting},
  author={Song, Zhangyao and Qu, Chaofeng and Zha, Chao and Zhao, Xiaoyu and Xu, Yinfei and Guo, Tao},
  year={2026}
}
```

## License

MIT
