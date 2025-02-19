from typing import Optional, List

import torch

from core.logger import Logger
from omni.isaac.core.articulations import ArticulationView
from torch import Tensor

from core.archiver import Archiver
from core.base import BaseObject
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
)
from core.universe import Universe
from core.utils.limits import dict_to_vec_limits


def apply_joint_position_limits(
    vec_joint_position_limits: VecJointPositionLimits,
    prim_paths: List[str],
    joint_names: List[str],
) -> None:
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdPhysics

    stage = get_current_stage()

    for _, prim_path in enumerate(prim_paths):
        for j, joint_name in enumerate(joint_names):
            joint_path = f"{prim_path}/{joint_name}"

            rev_joint = UsdPhysics.RevoluteJoint.Get(stage, joint_path)

            if not rev_joint:
                continue  # Skip if joint is not a RevoluteJoint

            # in degrees: https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr/UsdPhysics.html#pxr.UsdPhysics.RevoluteJoint
            rev_joint.CreateLowerLimitAttr().Set(
                vec_joint_position_limits[j, 0].item(),
            )
            rev_joint.GetUpperLimitAttr().Set(
                vec_joint_position_limits[j, 1].item(),
            )


class VecJointsController(BaseObject):
    def __init__(
        self,
        universe: Universe,
        noise_function: NoiseFunction,
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

        self.path_expr: str = ""

        self._articulation_view: Optional[ArticulationView] = None

        self._noise_function: NoiseFunction = noise_function
        self._target_joint_positions: Tensor = torch.zeros(
            (self._universe.num_envs, len(actuators))
        )  # Target positions in rads

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

    @property
    def art_view(self) -> Optional[ArticulationView]:
        if not self._is_constructed:
            return None

        return self._articulation_view

    def construct(self, path_expr: str) -> None:
        super().construct()

        self.path_expr = path_expr

        # zero out any fixed joints' limits
        fixed_joint_indices = (
            self._vec_fixed_joints.cpu().nonzero(as_tuple=True)[0].flatten()
        )

        self._vec_joint_position_limits[fixed_joint_indices] = 0.0
        self._vec_joint_position_limits_rad[fixed_joint_indices] = 0.0
        self._vec_joint_velocity_limits[fixed_joint_indices] = 0.0
        self._vec_joint_velocity_limits_rad[fixed_joint_indices] = 0.0
        self._vec_joint_effort_limits[fixed_joint_indices] = 0.0

        from omni.isaac.core.articulations import ArticulationView

        self._articulation_view = ArticulationView(
            self.path_expr,
            name="joints_controller_art_view",
            reset_xform_properties=False,
        )
        self._universe.add_prim(self._articulation_view)

        for i, actuator in enumerate(self._actuators):
            actuator.register_self(
                self._vec_joint_velocity_limits_rad[i],
                self._vec_joint_effort_limits[i],
                self._vec_gear_ratios[i],
            )

        Logger.info("JointsController constructed.")

        self._is_constructed = True

    def post_construct(self):
        super().post_construct()

        assert self._articulation_view.num_dof == self._num_joints, (
            f"Number of dof in articulation view ({self._articulation_view.num_dof}) "
            f"does not match the number of joints in the controller ({self._num_joints})"
        )

        assert self._articulation_view.dof_names == list(
            self._joint_position_limits.keys()
        ), (
            f"Joint names in articulation view ({self._articulation_view.dof_names}) "
            f"do not match the position limits' joint names ({list(self._joint_position_limits.keys())}; in order or "
            f"content)"
        )
        assert self._articulation_view.dof_names == list(
            self._joint_velocity_limits.keys()
        ), (
            f"Joint names in articulation view ({self._articulation_view.dof_names}) "
            f"do not match the velocity limits' joint names ({list(self._joint_velocity_limits.keys())}; in order or "
            f"content)"
        )
        assert self._articulation_view.dof_names == list(
            self._joint_effort_limits.keys()
        ), (
            f"Joint names in articulation view ({self._articulation_view.dof_names}) "
            f"do not match the effort limits' joint names ({list(self._joint_effort_limits.keys())}; in order or "
            f"content)"
        )
        assert self._articulation_view.dof_names == list(self._gear_ratios.keys()), (
            f"Joint names in articulation view ({self._articulation_view.dof_names}) "
            f"do not match the gear ratios' joint names ({list(self._gear_ratios.keys())}; in order or "
            f"content)"
        )
        assert self._articulation_view.dof_names == list(self._fixed_joints.keys()), (
            f"Joint names in articulation view ({self._articulation_view.dof_names}) "
            f"do not match the fixed joints' joint names ({list(self._fixed_joints.keys())}; in order or "
            f"content)"
        )

        apply_joint_position_limits(
            self._vec_joint_position_limits,
            self._articulation_view.prim_paths,
            self._articulation_view.joint_names,
        )

        self._is_post_constructed = True

    def step(self, joint_actions: Tensor) -> None:
        assert (
            self.is_fully_constructed
        ), "Joints controller not fully constructed: tried to step!"

        self._target_joint_positions = self._process_joint_actions(
            joint_actions,
            self._vec_joint_position_limits_rad,
            self._noise_function,
        )

        current_joint_positions = (
            self._articulation_view.get_joint_positions()
        )  # in radians
        current_velocities = (
            self._articulation_view.get_joint_velocities()
        )  # in radians per second

        efforts_to_apply: Tensor = torch.zeros_like(self._target_joint_positions)

        for i, actuator in enumerate(self._actuators):
            efforts = actuator.step(
                current_joint_positions[:, i],
                self._target_joint_positions[:, i],
                current_velocities[:, i],
            )
            efforts_to_apply[:, i] = efforts

        self._articulation_view.set_joint_efforts(efforts_to_apply)

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
        assert (
            self.is_fully_constructed
        ), "Joints controller not fully constructed: tried to reset!"

        if indices is None:
            indices = torch.arange(
                self._articulation_view.count,
                device=self._universe.device,
            )
        else:
            indices = indices.to(device=self._universe.device)

        joint_positions = joint_positions.to(device=self._universe.device)

        self._target_joint_positions = self._process_joint_actions(
            joint_positions,
            self._vec_joint_position_limits,
        )

        self._articulation_view.set_joint_positions(
            self._target_joint_positions,
            indices,
        )

        self._articulation_view.set_joint_velocities(
            joint_velocities,
            indices,
        )

        self._articulation_view.set_joint_efforts(
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
        return torch.rad2deg(self._articulation_view.get_joint_positions())

    def get_joint_velocities_deg(self) -> Tensor:
        return torch.rad2deg(self._articulation_view.get_joint_velocities())

    def get_joint_positions_rad(self) -> Tensor:
        return self._articulation_view.get_joint_positions()

    def get_joint_velocities_rad(self) -> Tensor:
        return self._articulation_view.get_joint_velocities()

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
