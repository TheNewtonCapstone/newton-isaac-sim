from typing import Optional

import torch
from core.types import IMUData, NoiseFunction
from core.universe import Universe
from omni.isaac.core.prims import RigidPrimView
from torch import Tensor

# TODO: Wrap sensors in a ROS2 node for easy integration
#   This would allow for easy logging of commands, sensor data, etc. all within the ROS2 ecosystem.


class VecIMU:
    def __init__(
        self,
        universe: Universe,
        local_position: Tensor,
        local_orientation: Tensor,
        noise_function: NoiseFunction,
    ):
        self._universe: Universe = universe
        self.path_expr: str = ""
        self.local_position: Tensor = local_position.to(self._universe.device)
        self.local_orientation: Tensor = local_orientation.to(self._universe.device)
        self.num_imus: int = 0

        self._rigid_prim_view: Optional[RigidPrimView] = None
        self._last_update_time: float = 0.0

        self._noise_function: NoiseFunction = noise_function
        self._is_constructed: bool = False

        from core.utils.math import IDENTITY_QUAT

        self._positions: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )
        self._rotations: Tensor = IDENTITY_QUAT.repeat(self.num_imus, 1).to(
            self._universe.device
        )

        self._linear_accelerations: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )
        self._linear_velocities: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )

        self._angular_accelerations: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )
        self._angular_velocities = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )

        self._last_linear_velocities: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )
        self._last_angular_velocities: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )

        self._projected_gravities: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )

    def construct(self, path_expr: str) -> None:
        assert not self._is_constructed, "IMU already constructed: tried to construct!"

        self.path_expr = path_expr

        self._rigid_prim_view = RigidPrimView(
            self.path_expr,
            name="imu_rigid_view",
            prepare_contact_sensors=False,
            disable_stablization=False,
            reset_xform_properties=False,
        )
        self._universe.add_prim(self._rigid_prim_view)

        # propagate physics changes
        self._universe.reset()

        self.num_imus = self._rigid_prim_view.count

        self._is_constructed = True

        # required to fill the tensors with the correct number of IMUs
        self.reset()

    def reset(self) -> None:
        from core.utils.math import IDENTITY_QUAT

        self._positions: Tensor = torch.zeros(
            (self.num_imus, 3), device=self._universe.device
        )
        self._rotations: Tensor = IDENTITY_QUAT.repeat(self.num_imus, 1).to(
            self._universe.device
        )

        self._linear_accelerations: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )
        self._linear_velocities: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )

        self._angular_accelerations: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )
        self._angular_velocities: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )

        self._last_linear_velocities: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )
        self._last_angular_velocities: Tensor = torch.zeros(
            (self.num_imus, 3),
            device=self._universe.device,
        )

        self._projected_gravities: Tensor = torch.zeros(
            (self.num_imus, 3),
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
        assert self._is_constructed, "IMU not constructed: tried to update data!"

        # from: https://github.com/isaac-sim/IsaacLab/pull/619/files#diff-44fe42c247de7301a3ce18a10d2b8c9045d58d42fc8440a7221b458d0712e83d

        physics_dt = self._universe.current_time - self._last_update_time
        self._last_update_time = self._universe.current_time

        positions, orientations = self._rigid_prim_view.get_world_poses()
        positions = torch.nan_to_num(positions)
        orientations = torch.nan_to_num(orientations)  # formatted as wxyz

        # get the offset from COM to local origin
        com_positions = self._rigid_prim_view.get_coms()[0].squeeze(1)

        # obtain the velocities of the COMs, we use get_velocities instead of independent calls because we're running
        # in a GPU pipeline
        velocities = self._rigid_prim_view.get_velocities()
        linear_velocities = velocities[:, :3]
        angular_velocities = velocities[:, 3:]

        from omni.isaac.core.utils.torch import (
            quat_rotate_inverse,
            quat_mul,
            quat_rotate,
            get_euler_xyz,
        )

        positions += quat_rotate(orientations, self.local_position)
        orientations = quat_mul(orientations, self.local_orientation)

        # if an offset is present of the COM does not agree with the local origin, the linear velocity has to be
        # transformed taking the angular velocity into account
        linear_velocities += torch.cross(
            angular_velocities,
            quat_rotate(orientations, self.local_position - com_positions),
            dim=-1,
        )

        if physics_dt == 0:
            return

        # numerical derivations
        linear_accelerations = (
            linear_velocities - self._last_linear_velocities
        ) / physics_dt
        angular_accelerations = (
            angular_velocities - self._last_angular_velocities
        ) / physics_dt

        gravity = self._universe.physics_sim_view.get_gravity()
        projected_gravities = torch.tensor(
            gravity,
            device=self._universe.device,
        ).repeat(self.num_imus, 1)

        # store pose
        self._positions = positions
        rolls, pitches, yaws = get_euler_xyz(orientations, True)
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
