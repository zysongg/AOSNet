"""
CycleNet: cycle-based forecasting with optional probabilistic head.

This file keeps the original point forecaster and adds `CycleNetProb`,
which wraps the CycleNet point path with a Gaussian output head for
probabilistic forecasting in TSFLib.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from tsflib.layers import CycleLayer


class DecompNet(nn.Module):
    """Decomposition Network for CycleNet."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        hidden_dim: int,
        ma_type: str = "ema",
        kernel: int = 25,
        stride: int = 1,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len

        self.model = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_len),
        )

    def forward(self, x):
        return self.model(x)


@dataclass
class CycleNetProbConfigs:
    lookback_len: int = 96
    horizon_len: int = 96
    num_features: int = 7
    d_model: int = 256
    dropout: float = 0.1
    cycle: int = 24
    model_type: str = "linear"
    pre_train_path: Optional[str] = None
    sigma_min: float = 1e-4
    num_samples: int = 100


class Model(nn.Module):
    """CycleNet point forecaster."""

    VALID_MODEL_TYPES = {"linear", "mlp", "dmlp"}

    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.seq_len = getattr(configs, "lookback_len", 96)
        self.pred_len = getattr(configs, "horizon_len", 96)
        self.enc_in = getattr(configs, "num_features", 7)
        self.d_model = getattr(configs, "d_model", 256)
        self.dropout = getattr(configs, "dropout", 0.1)
        self.cycle_len = getattr(configs, "cycle", 24)
        self.model_type = getattr(configs, "model_type", "linear")
        self.pre_train_path = getattr(configs, "pre_train_path", None)

        if self.model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {self.VALID_MODEL_TYPES}, "
                f"got {self.model_type}"
            )

        self.cycleQueue = CycleLayer(
            cycle_len=self.cycle_len,
            num_features=self.enc_in,
            pre_train_path=self.pre_train_path,
        )

        if self.model_type == "linear":
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == "mlp":
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len),
            )
        else:
            self.model = DecompNet(
                input_len=self.seq_len,
                output_len=self.pred_len,
                hidden_dim=self.d_model,
                ma_type="ema",
                kernel=25,
                stride=1,
                alpha=0.5,
                beta=0.5,
            )

    def point_forecast(self, batch_) -> torch.Tensor:
        """Point forecast with shape `(B, C, H)`."""
        x, batch_y, batch_x_mark, batch_y_mark, index = batch_

        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        cycle_index = index.long() % self.cycle_len
        cycle_values = self.cycleQueue(cycle_index, self.seq_len + self.pred_len)
        cycle_x = cycle_values[:, :, :self.seq_len]
        cycle_y = cycle_values[:, :, self.seq_len:]

        x_res = x - cycle_x
        y_res = self.model(x_res)
        return y_res + cycle_y

    def forward(self, batch_) -> torch.Tensor:
        return self.point_forecast(batch_)


class ProbModel(Model):
    """CycleNetProb: Gaussian probabilistic forecaster built on CycleNet."""

    def __init__(self, configs):
        super().__init__(configs)
        self.sigma_min = getattr(configs, "sigma_min", 1e-4)
        self.num_samples = getattr(configs, "num_samples", 100)
        self.out_linear_mu = nn.Linear(self.enc_in, self.enc_in)
        self.out_linear_sigma = nn.Linear(self.enc_in, self.enc_in)

    def _forecast_distribution(self, batch_) -> Normal:
        point_forecast = self.point_forecast(batch_)          # (B, C, H)
        point_forecast = point_forecast.transpose(1, 2)      # (B, H, C)

        mu = self.out_linear_mu(point_forecast).transpose(1, 2)
        sigma = self.out_linear_sigma(point_forecast)
        sigma = F.softplus(sigma).transpose(1, 2) + self.sigma_min

        return Normal(loc=mu, scale=sigma)

    def forward(self, batch_) -> Normal:
        return self._forecast_distribution(batch_)

    def train_loss(self, batch_) -> torch.Tensor:
        _, y, _, _, _ = batch_
        y = y.float()
        dist = self._forecast_distribution(batch_)
        return -dist.log_prob(y).mean()

    def val_loss(self, batch_) -> torch.Tensor:
        return self.train_loss(batch_)

    def test_loss(self, batch_) -> torch.Tensor:
        return self.train_loss(batch_)

    @torch.no_grad()
    def sample(self, batch_, num_samples: Optional[int] = None) -> torch.Tensor:
        """Return samples with shape `(B, S, C, H)`."""
        if num_samples is None:
            num_samples = self.num_samples

        dist = self._forecast_distribution(batch_)
        samples = dist.rsample((num_samples,))               # (S, B, C, H)
        return samples.permute(1, 0, 2, 3).contiguous()


class CycleNetProb:
    Model = ProbModel
    Configs = CycleNetProbConfigs


CycleNet = Model
CycleNetProbModel = ProbModel

__all__ = [
    "Model",
    "CycleNet",
    "CycleNetProb",
    "CycleNetProbModel",
    "CycleNetProbConfigs",
]
