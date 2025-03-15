import numpy as np
import torch


@torch.jit.script
def squared(
    a: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    return torch.square(a) * weight


@torch.jit.script
def dot(
    a: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    return torch.sum(squared(a), dim=1) * weight


@torch.jit.script
def one_minus_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    return (1.0 - torch.sum(a * b, dim=1)) * weight


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
def fd_first_order_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    fd = a - b
    sqn = dot(fd, weight)

    return sqn


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
def exp_one_minus_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    mult: float = 1.0,
    weight: float = 1.0,
) -> torch.Tensor:
    sqd = one_minus_dot(a, b)
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
def exp_fd_first_order_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    mult: float = 1.0,
    weight: float = 1.0,
) -> torch.Tensor:
    fd_sqn = fd_first_order_dot(a, b)
    weighted_exp = torch.exp(mult * fd_sqn) * weight

    return weighted_exp
