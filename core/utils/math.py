import random

import numpy as np
import torch
from torch import Tensor

IDENTITY_QUAT: Tensor = torch.tensor([1, 0, 0, 0], dtype=torch.float32)


def gaussian_distribute(value: torch.Tensor | np.ndarray, mu: float = 1.0,
                        sigma: float = 0.065) -> torch.Tensor | np.ndarray:
    # Gaussian distribution
    return value * random.gauss(mu, sigma)


def lerp(a: float | torch.Tensor | np.ndarray, b: float | torch.Tensor | np.ndarray, t: float) -> float | torch.Tensor | np.ndarray:
    # Linear interpolation
    return a + t * (b - a)


def quat_slerp_t(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    assert a.shape == b.shape, "Shapes must match"

    cos_half_theta = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if abs(cos_half_theta) >= 1.0:
        return a

    half_theta = torch.arccos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratio_a = torch.sin((1.0 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta

    qc = torch.zeros_like(a)
    qc[0] = a[0] * ratio_a + b[0] * ratio_b
    qc[1] = a[1] * ratio_a + b[1] * ratio_b
    qc[2] = a[2] * ratio_a + b[2] * ratio_b
    qc[3] = a[3] * ratio_a + b[3] * ratio_b

    return qc


def quat_slerp_n(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    assert a.shape == b.shape, "Shapes must match"

    cos_half_theta = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if abs(cos_half_theta) >= 1.0:
        return a

    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratio_a = np.sin((1.0 - t) * half_theta) / sin_half_theta
    ratio_b = np.sin(t * half_theta) / sin_half_theta

    qc = np.array(a)
    qc[0] = a[0] * ratio_a + b[0] * ratio_b
    qc[1] = a[1] * ratio_a + b[1] * ratio_b
    qc[2] = a[2] * ratio_a + b[2] * ratio_b
    qc[3] = a[3] * ratio_a + b[3] * ratio_b

    return qc


def difference_length_squared_n(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b

    return np.sum(diff * diff, axis=1)


def difference_length_n(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b

    return np.linalg.norm(diff)


def length_squared_n(a: np.ndarray) -> np.ndarray:
    return np.sum(a * a, axis=1)


def length_n(a: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a)
