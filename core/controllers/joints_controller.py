from typing import Optional, List

import torch
from genesis.engine.entities import RigidEntity
from torch import Tensor

from core.actuators import BaseActuator
from core.archiver import Archiver
from core.base import BaseObject
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

        self._noise_function: NoiseFunction = noise_function
        self._target_joint_positions: Tensor = torch.zeros(
            (self._universe.num_envs, len(actuators))
        )  # Target positions in rads

        self._joint_names: JointNames = joint_names
        self._joints_dof_idx: List[int] = []
        self._num_joints: int = len(joint_names)

        self._joint_position_limits: ArtJointsPositionLimits = joint_position_limits
        self._vec_joint_position_limits: VecJointPositionLimits = dict_to_vec_limits(
            joint_position_limits,
            self.device,
        )
        self._vec_joint_position_limits_rad: VecJointPositionLimits = torch.deg2rad(
            self._vec_joint_position_limits,
        )

        self._joint_velocity_limits: ArtJointsVelocityLimits = joint_velocity_limits
        self._vec_joint_velocity_limits: VecJointVelocityLimits = dict_to_vec_limits(
            joint_velocity_limits,
            self.device,
        )
        self._vec_joint_velocity_limits_rad: VecJointVelocityLimits = torch.deg2rad(
            self._vec_joint_velocity_limits,
        )

        self._joint_effort_limits: ArtJointsEffortLimits = joint_effort_limits
        self._vec_joint_effort_limits: VecJointEffortLimits = dict_to_vec_limits(
            joint_effort_limits,
            self.device,
        )

        self._gear_ratios: ArtJointsGearRatios = joint_gear_ratios
        self._vec_gear_ratios: VecJointGearRatios = dict_to_vec_limits(
            joint_gear_ratios,
            self.device,
        )

        self._fixed_joints: ArtJointsFixed = fixed_joints
        self._vec_fixed_joints: VecJointFixed = dict_to_vec_limits(
            fixed_joints,
            self.device,
        )

        self._actuators: List[BaseActuator] = actuators

    @property
    def joint_names(self) -> JointNames:
        return self._joint_names

    @property
    def target_joint_positions_deg(self) -> Tensor:
        return torch.rad2deg(self._target_joint_positions)

    @property
    def joint_positions_deg(self) -> Tensor:
        return torch.rad2deg(self.joint_positions_rad)

    @property
    def joint_velocities_deg(self) -> Tensor:
        return torch.rad2deg(self.joint_velocities_rad)

    @property
    def joint_positions_rad(self) -> Tensor:
        return self._robot.get_dofs_position(
            dofs_idx_local=self._joints_dof_idx,
        )

    @property
    def joint_velocities_rad(self) -> Tensor:
        return self._robot.get_dofs_velocity(
            dofs_idx_local=self._joints_dof_idx,
        )

    @property
    def applied_joint_efforts(self) -> Tensor:
        applied_joint_efforts: Tensor = torch.zeros_like(self._target_joint_positions)

        for i, actuator in enumerate(self._actuators):
            applied_joint_efforts[:, i] = actuator.applied_output_efforts.squeeze(-1)

        return applied_joint_efforts

    def pre_build(self) -> None:
        super().pre_build()

        # zero out any fixed joints' limits
        fixed_joint_indices = (
            self._vec_fixed_joints.cpu().nonzero(as_tuple=True)[0].flatten()
        )

        self._vec_joint_position_limits[fixed_joint_indices] = 0.0
        self._vec_joint_position_limits_rad[fixed_joint_indices] = 0.0
        self._vec_joint_velocity_limits[fixed_joint_indices] = 0.0
        self._vec_joint_velocity_limits_rad[fixed_joint_indices] = 0.0
        self._vec_joint_effort_limits[fixed_joint_indices] = 0.0

        for i, actuator in enumerate(self._actuators):
            pre_build_kwargs = {
                "output_vec_velocity_limits": self._vec_joint_velocity_limits_rad[i],
                "output_vec_effort_limits": self._vec_joint_effort_limits[i],
                "vec_gear_ratios": self._vec_gear_ratios[i],
            }
            actuator.register_self(pre_kwargs=pre_build_kwargs)

        self._is_pre_built = True

    def post_build(self, robot: RigidEntity) -> None:
        super().post_build()

        self._robot = robot

        self._joints_dof_idx = [
            self._robot.get_joint(name).dof_idx_local for name in self._joint_names
        ]

        self._robot.set_dofs_kp(
            [8.0] * self._num_joints,
            dofs_idx_local=self._joints_dof_idx,
        )
        self._robot.set_dofs_kv(
            [0.5] * self._num_joints,
            dofs_idx_local=self._joints_dof_idx,
        )

        self._is_post_built = True

    def step(self, joint_actions: Tensor) -> None:
        if self._noise_function is not None:
            joint_actions = self._noise_function(joint_actions)

        self._target_joint_positions = joint_actions.to(self.device)

        efforts_to_apply: Tensor = torch.zeros_like(self._target_joint_positions)

        # cache, to avoid multiple expensive calls in the loop
        joint_positions_rad = self.joint_positions_rad
        joint_velocities_rad = self.joint_velocities_rad

        for i, actuator in enumerate(self._actuators):
            efforts = actuator.step(
                joint_positions_rad[:, i],
                self._target_joint_positions[:, i],
                joint_velocities_rad[:, i],
            )
            efforts_to_apply[:, i] = efforts

        # self._robot.control_dofs_force(
        #    efforts_to_apply,
        #    dofs_idx_local=self._joints_dof_idx,
        # )

        self._robot.control_dofs_position(
            position=self._target_joint_positions,
            dofs_idx_local=self._joints_dof_idx,
        )

        if Archiver.enabled:
            joints_obs_archive = {
                "joint_positions": self.joint_positions_deg,
                "joint_velocities_median": self.joint_velocities_deg.median(),
                "joint_efforts_median": self.applied_joint_efforts.median(),
                "joint_efforts": self.applied_joint_efforts,
            }
            Archiver.put("joints_obs", joints_obs_archive)

    def reset(
        self,
        joint_positions: Optional[Tensor] = None,
        joint_velocities: Optional[Tensor] = None,
        joint_efforts: Optional[Tensor] = None,
        indices: Optional[Indices] = None,
    ) -> None:
        """
        Reset the joint positions, velocities and efforts. If any of the arguments are None, they are not reset.
        Args:
            joint_positions: Positions to reset the joints to (radians).
            joint_velocities: Velocities to reset the joints to (in radians/second).
            joint_efforts: Efforts to reset the joints to.
            indices: Indices of the environments to reset. If None, reset all environments.

        Returns:
            None

        """
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        else:
            indices = indices.to(device=self.device)

        if joint_positions is not None:
            self._target_joint_positions = joint_positions.to(self.device)

            self._robot.set_dofs_position(
                position=self._target_joint_positions,
                dofs_idx_local=self._joints_dof_idx,
                envs_idx=indices,
                zero_velocity=True,
            )

        if joint_velocities is not None:
            self._robot.set_dofs_velocity(
                velocity=joint_velocities,
                dofs_idx_local=self._joints_dof_idx,
                envs_idx=indices,
            )

        if joint_efforts is not None:
            self._robot.control_dofs_force(
                force=joint_efforts,
                dofs_idx_local=self._joints_dof_idx,
                envs_idx=indices,
            )
