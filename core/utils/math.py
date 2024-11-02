# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for various math operations."""

# needed to import for allowing type-hinting: torch.Tensor | np.ndarray
from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
import torch.nn.functional

"""
Sampling
"""


def gaussian_distribute(value: torch.Tensor, mu: float = 1.0, sigma: float = 0.065) -> torch.Tensor:
    # Gaussian distribution
    return value * random.gauss(mu, sigma)


"""
Rotation
"""


def convert_quat(quat: torch.Tensor | np.ndarray, to: Literal["xyzw", "wxyz"] = "xyzw") -> torch.Tensor | np.ndarray:
    """Converts quaternion from one convention to another.

    The convention to convert TO is specified as an optional argument. If to == 'xyzw',
    then the input is in 'wxyz' format, and vice-versa.

    Args:
        quat: The quaternion of shape (..., 4).
        to: Convention to convert quaternion to.. Defaults to "xyzw".

    Returns:
        The converted quaternion in specified convention.

    Raises:
        ValueError: Invalid input argument `to`, i.e. not "xyzw" or "wxyz".
        ValueError: Invalid shape of input `quat`, i.e. not (..., 4,).
    """
    # check input is correct
    if quat.shape[-1] != 4:
        msg = f"Expected input quaternion shape mismatch: {quat.shape} != (..., 4)."
        raise ValueError(msg)
    if to not in ["xyzw", "wxyz"]:
        msg = f"Expected input argument `to` to be 'xyzw' or 'wxyz'. Received: {to}."
        raise ValueError(msg)
    # check if input is numpy array (we support this backend since some classes use numpy)
    if isinstance(quat, np.ndarray):
        # use numpy functions
        if to == "xyzw":
            # wxyz -> xyzw
            return np.roll(quat, -1, axis=-1)
        else:
            # xyzw -> wxyz
            return np.roll(quat, 1, axis=-1)
    else:
        # convert to torch (sanity check)
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=float)
        # convert to specified quaternion type
        if to == "xyzw":
            # wxyz -> xyzw
            return quat.roll(-1, dims=-1)
        else:
            # xyzw -> wxyz
            return quat.roll(1, dims=-1)


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (N, 4).
        v: The vector in (x, y, z). Shape is (N, 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (N, 3).
    """
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c
