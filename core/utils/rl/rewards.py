import numpy as np
import torch


@torch.jit.script
def squared_norm(
    a: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    return (torch.linalg.vector_norm(a, dim=1, keepdim=False) ** 2) * weight


@torch.jit.script
def squared_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    return (torch.sum(a * b, dim=1) ** 2) * weight


@torch.jit.script
def one_minus_squared_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    return (1.0 - torch.sum(a * b, dim=1) ** 2) * weight


@torch.jit.script
def squared(
    a: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    return (a**2) * weight


@torch.jit.script
def fd_first_order_squared(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    fd = a - b
    sq = squared(fd, weight)

    return sq


@torch.jit.script
def fd_first_order_abs(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    fd = a - b
    abs_fd = torch.abs(fd) * weight

    return abs_fd


@torch.jit.script
def fd_first_order_squared_norm(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    fd = a - b
    sqn = squared_norm(fd, weight)

    return sqn


@torch.jit.script
def fd_second_order_squared_norm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    fd = a - 2 * b + c
    sqn = squared_norm(fd, weight)

    return sqn


@torch.jit.script
def exp_squared_norm(
    a: torch.Tensor,
    mult: float = 1.0,
    weight: float = 1.0,
) -> torch.Tensor:
    sqn = squared_norm(a)
    weighted_exp = torch.exp(mult * sqn) * weight

    return weighted_exp


@torch.jit.script
def fd_first_order_sum_abs(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    sum = torch.sum(torch.abs(a - b), dim=1)
    weighted_sum = sum * weight

    return weighted_sum


@torch.jit.script
def exp_one_minus_squared_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    mult: float = 1.0,
    weight: float = 1.0,
) -> torch.Tensor:
    sqd = one_minus_squared_dot(a, b)
    weighted_exp = torch.exp(mult * sqd) * weight

    return weighted_exp


@torch.jit.script
def exp_squared(
    a: torch.Tensor,
    mult: float = 1.0,
    weight: float = 1.0,
) -> torch.Tensor:
    sq = squared(a)
    weighted_exp = torch.exp(mult * sq) * weight

    return weighted_exp


@torch.jit.script
def exp_fd_first_order_squared_norm(
    a: torch.Tensor,
    b: torch.Tensor,
    mult: float = 1.0,
    weight: float = 1.0,
) -> torch.Tensor:
    fd_sqn = fd_first_order_squared_norm(a, b)
    weighted_exp = torch.exp(mult * fd_sqn) * weight

    return weighted_exp


@torch.jit.script
def exp_fd_second_order_squared_norm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    mult: float = 1.0,
    weight: float = 1.0,
) -> torch.Tensor:
    fd_sqn = fd_second_order_squared_norm(a, b, c)
    weighted_exp = torch.exp(mult * fd_sqn) * weight

    return weighted_exp


def kl_based_adaptive_lr(
    progress_remaining: float,
    current_lr: float,
    current_kl: float,
    target_kl: float,
    start_lr: float = 1e-3,
) -> float:
    if target_kl is None:
        return start_lr  # starting lr

    if isinstance(target_kl, np.ndarray):
        target_kl = target_kl.item()

    if current_kl > 2.0 * target_kl:
        return max(10e-5, current_lr / 1.5)

    if current_kl < 0.5 * target_kl:
        return min(10e-2, current_lr * 1.5)

    return current_lr
