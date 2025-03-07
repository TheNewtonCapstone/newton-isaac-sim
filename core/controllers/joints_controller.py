from typing import Optional, List

import genesis as gs
from genesis.engine.entities import RigidEntity

import torch
from torch import Tensor

from core.actuators import BaseActuator
from core.archiver import Archiver
from core.base import BaseObject
from core.logger import Logger
from core.types import (
    NoiseFunction,
    Indices,
    ArtJointsPositionLimits,
    VecJointPositionLimits,
    ArtJointsVelocityLimits,
    ArtJointsEffortLimits,
    VecJointVelocityLimits,
    VecJointEffortLimits,
    ArtJointsGearRatios,
    VecJointGearRatios,
    ArtJointsFixed,
    VecJointFixed,
    JointNames,
)
from core.universe import Universe
from core.utils.limits import dict_to_vec_limits


class VecJointsController(BaseObject):
    def __init__(
        self,
        universe: Universe,
        num_envs: int,
        noise_function: NoiseFunction,
        joint_names: JointNames,
        joint_position_limits: ArtJointsPositionLimits,
        joint_velocity_limits: ArtJointsVelocityLimits,
        joint_effort_limits: ArtJointsEffortLimits,
        joint_gear_ratios: ArtJointsGearRatios,
        actuators: List[BaseActuator],
        fixed_joints: ArtJointsFixed,
    ):
        super().__init__(universe=universe)

        # We type hint universe again here to avoid circular imports
        self._universe: Universe = universe
        self._robot: Optional[RigidEntity] = None
        self._num_envs: int = num_envs

        self._noise_function: NoiseFunction = noise_function
        self._target_joint_positions: Tensor = torch.zeros(
            (self._universe.num_envs, len(actuators))
        )  # Target positions in rads

        self._joint_names: JointNames = joint_names
        self._joints_dof_idx: List[int] = []
        self._num_joints: int = len(joint_position_limits)

        self._joint_position_limits: ArtJointsPositionLimits = joint_position_limits
        self._vec_joint_position_limits: VecJointPositionLimits = dict_to_vec_limits(
            joint_position_limits,
            self._universe.device,
        )
        self._vec_joint_position_limits_rad: VecJointPositionLimits = torch.deg2rad(
            self._vec_joint_position_limits,
        )

        self._joint_velocity_limits: ArtJointsVelocityLimits = joint_velocity_limits
        self._vec_joint_velocity_limits: VecJointVelocityLimits = dict_to_vec_limits(
            joint_velocity_limits,
            self._universe.device,
        )
        self._vec_joint_velocity_limits_rad: VecJointVelocityLimits = torch.deg2rad(
            self._vec_joint_velocity_limits,
        )

        self._joint_effort_limits: ArtJointsEffortLimits = joint_effort_limits
        self._vec_joint_effort_limits: VecJointEffortLimits = dict_to_vec_limits(
            joint_effort_limits,
            self._universe.device,
        )

        self._gear_ratios: ArtJointsGearRatios = joint_gear_ratios
        self._vec_gear_ratios: VecJointGearRatios = dict_to_vec_limits(
            joint_gear_ratios,
            self._universe.device,
        )

        self._fixed_joints: ArtJointsFixed = fixed_joints
        self._vec_fixed_joints: VecJointFixed = dict_to_vec_limits(
            fixed_joints,
            self._universe.device,
        )

        self._actuators: List[BaseActuator] = actuators

    def post_build(self, robot: RigidEntity) -> None:
        # zero out any fixed joints' limits
        fixed_joint_indices = (
            self._vec_fixed_joints.cpu().nonzero(as_tuple=True)[0].flatten()
        )

        self._vec_joint_position_limits[fixed_joint_indices] = 0.0
        self._vec_joint_position_limits_rad[fixed_joint_indices] = 0.0
        self._vec_joint_velocity_limits[fixed_joint_indices] = 0.0
        self._vec_joint_velocity_limits_rad[fixed_joint_indices] = 0.0
        self._vec_joint_effort_limits[fixed_joint_indices] = 0.0

        self._robot = robot

        self._joints_dof_idx = [
            self._robot.get_joint(name).dof_idx_local for name in self._joint_names
        ]

        for i, actuator in enumerate(self._actuators):
            actuator.build(
                self._vec_joint_velocity_limits_rad[i],
                self._vec_joint_effort_limits[i],
                self._vec_gear_ratios[i],
            )

    def step(self, joint_actions: Tensor) -> None:
        self._target_joint_positions = self._process_joint_actions(
            joint_actions,
            self._vec_joint_position_limits_rad,
            self._noise_function,
        )

        current_joint_positions = self.get_joint_positions_rad()
        current_velocities = self.get_joint_velocities_rad()

        efforts_to_apply: Tensor = torch.zeros_like(self._target_joint_positions)

        for i, actuator in enumerate(self._actuators):
            efforts = actuator.step(
                current_joint_positions[:, i],
                self._target_joint_positions[:, i],
                current_velocities[:, i],
            )
            efforts_to_apply[:, i] = efforts

        self._robot.control_dofs_force(efforts_to_apply)

        joints_obs_archive = {
            "joint_positions_norm": self.get_normalized_joint_positions(),
            "joint_positions": self.get_joint_positions_deg(),
            "joint_velocities_norm_median": self.get_normalized_joint_velocities().median(),
            "joint_velocities_median": self.get_joint_velocities_deg().median(),
            "joint_efforts_median": self.get_applied_joint_efforts().median(),
            "joint_efforts": self.get_applied_joint_efforts(),
        }
        Archiver.put("joints_obs", joints_obs_archive)

    def reset(
        self,
        joint_positions: Tensor,
        joint_velocities: Tensor,
        joint_efforts: Tensor,
        indices: Optional[Indices] = None,
    ) -> None:
        if indices is None:
            indices = torch.arange(
                self._num_envs,
                device=self._universe.device,
            )
        else:
            indices = indices.to(device=self._universe.device)

        joint_positions = joint_positions.to(device=self._universe.device)

        self._target_joint_positions = self._process_joint_actions(
            joint_positions,
            self._vec_joint_position_limits,
        )

        self._robot.set_dofs_position(
            self._target_joint_positions,
            indices,
            zero_velocity=False,
        )

        self._robot.set_dofs_velocity(
            joint_velocities,
            indices,
        )

        self._robot.control_dofs_force(
            joint_efforts,
            indices,
        )

    def normalize_joint_positions(self, joint_positions: Tensor) -> Tensor:
        """
        Args:
            joint_positions: The joint positions to be normalized (in degrees).

        Returns:
            The normalized joint positions.
        """
        from core.utils.math import map_range

        joint_positions_normalized = map_range(
            joint_positions,
            self._vec_joint_position_limits[:, 0].to(
                joint_positions.device,
            ),
            self._vec_joint_position_limits[:, 1].to(
                joint_positions.device,
            ),
            -1.0,
            1.0,
        )

        return joint_positions_normalized

    def normalize_joint_velocities(self, joint_velocities: Tensor) -> Tensor:
        """
        Args:
            joint_velocities: The joint velocities to be normalized (in degrees).

        Returns:
            The normalized joint velocities.
        """
        from core.utils.math import map_range

        joint_velocities_normalized = map_range(
            joint_velocities,
            -self._vec_joint_velocity_limits.to(
                joint_velocities.device,
            ).squeeze(-1),
            self._vec_joint_velocity_limits.to(
                joint_velocities.device,
            ).squeeze(-1),
            -1.0,
            1.0,
        )

        return joint_velocities_normalized

    # TODO: Improve naming of `JointsController` methods
    #   They are too long and too descriptive, which makes them hard to read.

    def normalize_joint_efforts(self, joint_efforts: Tensor) -> Tensor:
        """
        Args:
            joint_efforts: The joint efforts to be normalized.

        Returns:
            The normalized joint efforts.
        """
        from core.utils.math import map_range

        joint_efforts_normalized = map_range(
            joint_efforts,
            -self._vec_joint_effort_limits.to(
                joint_efforts.device,
            ).squeeze(-1),
            self._vec_joint_effort_limits.to(
                joint_efforts.device,
            ).squeeze(-1),
            -1.0,
            1.0,
        )

        return joint_efforts_normalized

    def get_normalized_joint_positions(self) -> Tensor:
        """
        Returns:
            The joint positions normalized to the joint constraints [-1, 1].
        """
        return self.normalize_joint_positions(self.get_joint_positions_deg())

    def get_normalized_joint_velocities(self) -> Tensor:
        """
        Returns:
            The joint velocities normalized to the joint constraints [-1, 1].
        """

        return self.normalize_joint_velocities(self.get_joint_velocities_deg())

    def get_normalized_joint_efforts(self) -> Tensor:
        """
        Returns:
            The joint efforts normalized to the joint constraints [-1, 1].
        """
        return self.normalize_joint_efforts(self.get_applied_joint_efforts())

    def get_target_joint_positions_deg(self) -> Tensor:
        return torch.rad2deg(self._target_joint_positions)

    def get_joint_positions_deg(self) -> Tensor:
        return torch.rad2deg(self._robot.get_dofs_position())

    def get_joint_velocities_deg(self) -> Tensor:
        return torch.rad2deg(self._robot.get_dofs_velocity())

    def get_joint_positions_rad(self) -> Tensor:
        return self._robot.get_dofs_position()

    def get_joint_velocities_rad(self) -> Tensor:
        return self._robot.get_dofs_velocity()

    def get_applied_joint_efforts(self) -> Tensor:
        applied_joint_efforts: Tensor = torch.zeros_like(self._target_joint_positions)

        for i, actuator in enumerate(self._actuators):
            applied_joint_efforts[:, i] = actuator.applied_output_efforts.squeeze(-1)

        return applied_joint_efforts

    def _process_joint_actions(
        self,
        joint_actions: Tensor,
        vec_joint_position_limits: VecJointPositionLimits,
        noise_function: Optional[NoiseFunction] = None,
    ) -> Tensor:
        """
        Joint actions are processed by mapping them to the joint constraints (any unit) and applying noise.
        Args:
            joint_actions: The joint actions to be processed [-1, 1].
            vec_joint_position_limits: The joint position limits.
            noise_function: The noise function to be applied to the computed joint positions.

        Returns:
            The processed joint positions (in degrees).
        """
        joint_positions = torch.clamp(
            joint_actions.to(
                vec_joint_position_limits.device, dtype=vec_joint_position_limits.dtype
            ),
            min=-1.0,
            max=1.0,
        )

        joint_positions = torch.lerp(
            vec_joint_position_limits[:, 0],
            vec_joint_position_limits[:, 1],
            (joint_positions + 1) / 2,
        )

        if noise_function is not None:
            joint_positions = noise_function(joint_positions)

        return joint_positions
