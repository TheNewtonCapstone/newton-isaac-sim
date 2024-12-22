import torch


@torch.jit.script
def squared_norm(a: torch.Tensor, weight: float = 1.0, ) -> torch.Tensor:
    return (torch.linalg.vector_norm(a, dim=1, keepdim=False) ** 2) * weight


@torch.jit.script
def squared_dot(a: torch.Tensor, b: torch.Tensor, weight: float = 1.0, ) -> torch.Tensor:
    return (torch.sum(a * b, dim=1) ** 2) * weight


@torch.jit.script
def squared(a: torch.Tensor, weight: float = 1.0, ) -> torch.Tensor:
    return (a ** 2) * weight


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
def exp_squared_norm(a: torch.Tensor, mult: float = 1.0, weight: float = 1.0, ) -> torch.Tensor:
    sqn = squared_norm(a)
    weighted_exp = torch.exp(mult * sqn) * weight

    return weighted_exp


@torch.jit.script
def exp_squared_dot(a: torch.Tensor, b: torch.Tensor, mult: float = 1.0, weight: float = 1.0, ) -> torch.Tensor:
    sqd = squared_dot(a, b)
    weighted_exp = torch.exp(mult * sqd) * weight

    return weighted_exp


@torch.jit.script
def exp_squared(a: torch.Tensor, mult: float = 1.0, weight: float = 1.0, ) -> torch.Tensor:
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
