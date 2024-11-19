from __future__ import annotations

import random

import torch
from torch import Tensor

IDENTITY_QUAT: Tensor = torch.tensor([1, 0, 0, 0], dtype=torch.float32)


def gaussian_distribute(value: torch.Tensor, mu: float = 1.0, sigma: float = 0.065) -> torch.Tensor:
    # Gaussian distribution
    return value * random.gauss(mu, sigma)
