"""TSFLib Models — AOSNet open-source release."""

from typing import Dict, List, Type, Any, Optional
import torch.nn as nn

from tsflib.models.base import ModelBase
from tsflib.models.mlp import AOSNet, AOSNetModel, CycleNet, CycleNetModel, TQNet, TQNetModel

POINT_MODELS: Dict[str, Type] = {
    "AOSNet": AOSNet,
    "CycleNet": CycleNet,
    "TQNet": TQNet,
}

PROB_MODELS: Dict[str, Type] = {}
ALL_MODELS: Dict[str, Type] = {**POINT_MODELS, **PROB_MODELS}

def list_models(model_type: Optional[str] = None) -> List[str]:
    if model_type == "point":
        return sorted(POINT_MODELS.keys())
    elif model_type == "prob":
        return sorted(PROB_MODELS.keys())
    else:
        return sorted(ALL_MODELS.keys())

def get_model(name: str) -> Type:
    if name not in ALL_MODELS:
        available = ", ".join(sorted(ALL_MODELS.keys()))
        raise ValueError(f"Model '{name}' not found. Available: {available}")
    return ALL_MODELS[name]

def create_model(name: str, configs: Any) -> nn.Module:
    model_class = get_model(name)
    if hasattr(model_class, "Model"):
        return model_class.Model(configs)
    return model_class(configs)

__all__ = [
    "ModelBase",
    "AOSNet", "AOSNetModel",
    "CycleNet", "CycleNetModel",
    "TQNet", "TQNetModel",
    "list_models", "get_model", "create_model",
    "POINT_MODELS", "PROB_MODELS", "ALL_MODELS",
]
