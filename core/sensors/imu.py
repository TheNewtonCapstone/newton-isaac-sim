from typing import Optional

import torch
from genesis.engine.entities import RigidEntity
from torch import Tensor

from ..base import BaseObject
from ..logger import Logger
from ..types import IMUData, NoiseFunction
from ..universe import Universe


class VecIMU(BaseObject):
    def __init__(
        self,
        universe: Universe,
        num_envs: int,
        local_position: Tensor,
        local_orientation: Tensor,
        noise_function: NoiseFunction,
    ):
        super().__init__(universe=universe)

        # We type hint universe again here to avoid circular imports
        self._universe: Universe = universe

        self.local_position: Tensor = local_position.to(self._universe.device)
        self.local_orientation: Tensor = local_orientation.to(self._universe.device)
        self._num_envs: int = num_envs

        self._robot: Optional[RigidEntity] = None
        self._last_update_time: float = 0.0

        self._noise_function: NoiseFunction = noise_function

        from core.utils.math import IDENTITY_QUAT

        self._positions: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )
        self._rotations: Tensor = IDENTITY_QUAT.repeat(self._num_envs, 1).to(
            self._universe.device
        )

        self._linear_accelerations: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )
        self._linear_velocities: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )

        self._angular_accelerations: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )
        self._angular_velocities = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )

        self._last_linear_velocities: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )
        self._last_angular_velocities: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )

        self._projected_gravities: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )

    def pre_build(self) -> None:
        super().pre_build()

        self._is_pre_built = True

    def post_build(self, robot: RigidEntity) -> None:
        super().post_build()

        self._robot = robot

        # required to fill the tensors with the correct number of IMUs
        self.reset()

        self._is_post_built = True

    def reset(self) -> None:
        from core.utils.math import IDENTITY_QUAT

        self._positions: Tensor = torch.zeros(
            (self._num_envs, 3), device=self._universe.device
        )
        self._rotations: Tensor = IDENTITY_QUAT.repeat(self._num_envs, 1).to(
            self._universe.device
        )

        self._linear_accelerations: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )
        self._linear_velocities: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )

        self._angular_accelerations: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )
        self._angular_velocities: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )

        self._last_linear_velocities: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )
        self._last_angular_velocities: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )

        self._projected_gravities: Tensor = torch.zeros(
            (self._num_envs, 3),
            device=self._universe.device,
        )

    def get_data(self) -> IMUData:
        raw_data = self.get_raw_data()

        for key, value in raw_data.items():
            raw_data[key] = self._noise_function(value)

        return raw_data

    def get_raw_data(self) -> IMUData:
        self._update_data()

        return {
            "positions": self._positions,
            "rotations": self._rotations,
            "linear_velocities": self._linear_velocities,
            "linear_accelerations": self._linear_accelerations,
            "angular_accelerations": self._angular_accelerations,
            "angular_velocities": self._angular_velocities,
            "projected_gravities": self._projected_gravities,
        }

    def _update_data(self) -> None:
        # from: https://github.com/isaac-sim/IsaacLab/pull/619/files#diff-44fe42c247de7301a3ce18a10d2b8c9045d58d42fc8440a7221b458d0712e83d

        update_dt = self._universe.current_time - self._last_update_time
        self._last_update_time = self._universe.current_time

        positions = self._robot.get_pos()
        orientations = self._robot.get_quat()

        linear_velocities = self._robot.get_vel()
        angular_velocities = self._robot.get_ang()

        positions += quat_rotate(orientations, self.local_position)
        orientations = quat_mul(orientations, self.local_orientation)

        # if an offset is present of the COM does not agree with the local origin, the linear velocity has to be
        # transformed taking the angular velocity into account
        linear_velocities += torch.cross(
            angular_velocities,
            quat_rotate(orientations, self.local_position - com_positions),
            dim=-1,
        )

        if update_dt == 0:
            return

        # numerical derivations
        linear_accelerations = (
            linear_velocities - self._last_linear_velocities
        ) / update_dt
        angular_accelerations = (
            angular_velocities - self._last_angular_velocities
        ) / update_dt

        gravity = self._universe.physics_sim_view.get_gravity()
        projected_gravities = torch.tensor(
            gravity,
            device=self._universe.device,
        ).repeat(self._num_envs, 1)

        # store pose
        self._positions = positions

        rolls, pitches, yaws = get_euler_xyz(orientations, True)
        rolls = torch.rad2deg(rolls)
        pitches = torch.rad2deg(pitches)
        yaws = torch.rad2deg(yaws)
        self._rotations = torch.stack([rolls, pitches, yaws], dim=-1)

        # store velocities
        self._linear_velocities = quat_rotate_inverse(orientations, linear_velocities)
        self._angular_velocities = quat_rotate_inverse(orientations, angular_velocities)

        # store accelerations
        self._linear_accelerations = quat_rotate_inverse(
            orientations,
            linear_accelerations,
        )
        self._angular_accelerations = quat_rotate_inverse(
            orientations,
            angular_accelerations,
        )

        self._last_linear_velocities = linear_velocities.clone()
        self._last_angular_velocities = angular_velocities.clone()

        self._projected_gravities = quat_rotate_inverse(
            orientations, projected_gravities
        )
        self._projected_gravities /= torch.norm(
            self._projected_gravities, dim=-1, keepdim=True
        )
