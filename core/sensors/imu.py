from typing import Optional

import numpy as np
from torch import Tensor

import torch
from core.types import IMUData, NoiseFunction
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrimView


class VecIMU:
    def __init__(
        self,
        path_expr: str,
        world: World,
        local_position: Tensor,
        local_rotation: Tensor,
        noise_function: NoiseFunction,
    ):
        self.world: World = world
        self.path_expr: str = path_expr
        self.local_position: Tensor = local_position.to(self.world.device)
        self.local_rotation: Tensor = local_rotation.to(self.world.device)
        self.num_imus: int = 0

        self.rigid_view: Optional[RigidPrimView] = None

        self._noise_function: NoiseFunction = noise_function
        self._is_constructed: bool = False

        from core.utils.math import IDENTITY_QUAT

        self._positions: Tensor = torch.zeros((self.num_imus, 3))
        self._rotations: Tensor = IDENTITY_QUAT.repeat(self.num_imus, 1)

        self._linear_accelerations: Tensor = torch.zeros((self.num_imus, 3))
        self._linear_velocities: Tensor = torch.zeros((self.num_imus, 3))

        self._angular_accelerations: Tensor = torch.zeros((self.num_imus, 3))
        self._angular_velocities = torch.zeros((self.num_imus, 3))

        self._last_linear_velocities: Tensor = torch.zeros((self.num_imus, 3))
        self._last_angular_velocities: Tensor = torch.zeros((self.num_imus, 3))

        self._projected_gravities: Tensor = torch.zeros((self.num_imus, 3))

    def __del__(self):
        pass

    def construct(self) -> None:
        if self._is_constructed:
            return

        self.rigid_view = RigidPrimView(self.path_expr)
        self.world.scene.add(self.rigid_view)

        self.world.reset()

        self.num_imus = self.rigid_view.count

        self.reset()

        self._is_constructed = True

    def update(self) -> IMUData:
        if self._is_constructed:
            self._update_data()

        return self.get_data()

    def reset(self) -> IMUData:
        from core.utils.math import IDENTITY_QUAT

        self._rotations = IDENTITY_QUAT.repeat(self.num_imus, 1).to(self.world.device)

        self._linear_accelerations = torch.zeros(
            (self.num_imus, 3), device=self.world.device
        )
        self._linear_velocities = torch.zeros(
            (self.num_imus, 3), device=self.world.device
        )
        self._angular_accelerations = torch.zeros(
            (self.num_imus, 3), device=self.world.device
        )
        self._angular_velocities = torch.zeros(
            (self.num_imus, 3), device=self.world.device
        )

        self._last_linear_velocities = torch.zeros(
            (self.num_imus, 3), device=self.world.device
        )
        self._last_angular_velocities = torch.zeros(
            (self.num_imus, 3), device=self.world.device
        )

        if self._is_constructed:
            self._update_data()

        return self.get_data()

    def get_data(self, recalculate: bool = False) -> IMUData:
        raw_data = self.get_raw_data(recalculate)

        for key, value in raw_data.items():
            raw_data[key] = self._noise_function(value)

        return raw_data

    def get_raw_data(self, recalculate: bool = False) -> IMUData:
        if recalculate:
            self._update_data()

        return {
            "positions": self._positions,
            "rotations": self._rotations,
            "linear_accelerations": self._linear_accelerations,
            "angular_velocities": self._angular_velocities,
            "projected_gravities": self._projected_gravities,
        }

    def _update_data(self) -> None:
        if not self._is_constructed:
            return

        # from: https://github.com/isaac-sim/IsaacLab/pull/619/files#diff-44fe42c247de7301a3ce18a10d2b8c9045d58d42fc8440a7221b458d0712e83d

        positions, rotations = self.rigid_view.get_world_poses()
        positions = torch.nan_to_num(positions)
        rotations = torch.nan_to_num(rotations)

        # get the offset from COM to local origin
        com_positions = self.rigid_view.get_coms()[0].squeeze(1)

        # obtain the velocities of the COMs
        linear_velocities = self.rigid_view.get_linear_velocities()
        angular_velocities = self.rigid_view.get_angular_velocities()

        from omni.isaac.core.utils.torch import (
            quat_rotate_inverse,
            quat_mul,
            quat_rotate,
            get_euler_xyz,
        )

        # if an offset is present of the COM does not agree with the local origin, the linera velocity has to be
        # transformed taking the angular velocity into account
        linear_velocities += torch.cross(
            angular_velocities,
            quat_rotate(rotations, self.local_position - com_positions),
            dim=-1,
        )

        # numerical derivations
        linear_accelerations = (
            linear_velocities - self._last_linear_velocities
        ) / self.world.get_physics_dt()
        angular_accelerations = (
            angular_velocities - self._last_angular_velocities
        ) / self.world.get_physics_dt()

        gravity = self.world.physics_sim_view.get_gravity()
        projected_gravities = torch.tensor(
            gravity,
            device=self.world.device,
        ).repeat(self.num_imus, 1)

        # store pose
        self._positions = positions + quat_rotate(rotations, self.local_position)
        rolls, pitches, yaws = get_euler_xyz(
            quat_mul(rotations, self.local_rotation),
            True,
        )
        self._rotations = torch.stack([rolls, pitches, yaws], dim=-1)

        # store velocities
        self._linear_velocities = quat_rotate_inverse(rotations, linear_velocities)
        self._angular_velocities = quat_rotate_inverse(rotations, angular_velocities)

        # store accelerations
        self._linear_accelerations = quat_rotate_inverse(
            rotations,
            quat_rotate_inverse(rotations, linear_accelerations),
        )
        self._angular_accelerations = quat_rotate_inverse(
            rotations,
            quat_rotate_inverse(rotations, angular_accelerations),
        )

        self._last_linear_velocities = linear_velocities.clone()
        self._last_angular_velocities = angular_velocities.clone()

        self._projected_gravities = quat_rotate_inverse(rotations, projected_gravities)
