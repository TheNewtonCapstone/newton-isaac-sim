import random

import numpy as np

import torch
from torch import Tensor

IDENTITY_QUAT: Tensor = torch.tensor([1, 0, 0, 0], dtype=torch.float32)


def gaussian_distribute(
    value: torch.Tensor | np.ndarray, mu: float = 1.0, sigma: float = 0.065
) -> torch.Tensor | np.ndarray:
    # Gaussian distribution
    return value * random.gauss(mu, sigma)


def map_range(
    value: float | torch.Tensor | np.ndarray,
    from_min: float | torch.Tensor | np.ndarray,
    from_max: float | torch.Tensor | np.ndarray,
    to_min: float | torch.Tensor | np.ndarray,
    to_max: float | torch.Tensor | np.ndarray,
) -> float | torch.Tensor | np.ndarray:
    # Map value from one range to another
    return (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min


def lerp(
    a: float | torch.Tensor | np.ndarray, b: float | torch.Tensor | np.ndarray, t: float
) -> float | torch.Tensor | np.ndarray:
    # Linear interpolation
    return a + t * (b - a)


def quat_to_euler_t(q: torch.Tensor) -> torch.Tensor:
    # Extract quaternion components
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q0 * q2 - q3 * q1)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(np.pi / 2, device=q.device),
        torch.asin(sinp),
    )

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Stack the results along a new last dimension
    return torch.stack([roll, pitch, yaw], dim=-1)


def quat_rotate_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    s = q[..., 0]
    u = q[..., 1:4]

    u_dot_v = torch.sum(u * v, dim=-1, keepdim=True)

    u_norm_squared = torch.sum(u * u, dim=-1, keepdim=True)

    cross = torch.cross(u, v, dim=-1)

    return (
        v * (s.unsqueeze(-1) ** 2 - u_norm_squared)
        + 2.0 * u_dot_v * u
        + 2.0 * s.unsqueeze(-1) * cross
    )


def quat_inverse_t(q: torch.Tensor) -> torch.Tensor:
    ns = torch.sum(q * q, dim=-1, keepdim=True)
    conjugate = quat_conjugate_t(q)
    return conjugate / ns


def quat_rotate_inverse_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate_t(quat_inverse_t(q), v)


def quat_mult_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Extract components
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    # Create output tensor with same batch dimensions
    batch_shape = a.shape[:-1]
    q = torch.zeros((*batch_shape, 4), device=a.device)

    # Compute quaternion multiplication
    q[..., 0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
    q[..., 1] = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2
    q[..., 2] = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1
    q[..., 3] = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0

    return q


def quat_conjugate_t(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)


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
