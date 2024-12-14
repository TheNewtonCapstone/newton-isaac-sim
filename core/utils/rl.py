import numpy as np


def squared_norm(a: np.ndarray, weight: float = 1.0) -> np.ndarray:
    return np.sum(a ** 2, axis=-1) * weight


def fd_first_order_squared_norm(
        a: np.ndarray,
        b: np.ndarray,
        weight: float = 1.0,
) -> np.ndarray:
    return squared_norm(a - b, weight)


def fd_second_order_squared_norm(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        weight: float = 1.0,
) -> np.ndarray:
    return squared_norm(a - 2 * b + c, weight)


def exp_squared_norm(a: np.ndarray, mult: float = 1.0, weight: float = 1.0) -> np.ndarray:
    return np.exp(squared_norm(a), mult) * weight


def exp_fd_first_order_squared_norm(
        a: np.ndarray,
        b: np.ndarray,
        mult: float = 1.0,
        weight: float = 1.0,
) -> np.ndarray:
    return np.exp(mult * fd_first_order_squared_norm(a, b)) * weight


def exp_fd_second_order_squared_norm(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        mult: float = 1.0,
        weight: float = 1.0,
) -> np.ndarray:
    return np.exp(mult * fd_second_order_squared_norm(a, b, c)) * weight
