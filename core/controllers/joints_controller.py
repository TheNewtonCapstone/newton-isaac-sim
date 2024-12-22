from typing import Optional, List

import numpy as np
import torch
from core.types import NoiseFunction, Indices, JointsConstraints
from core.universe import Universe
from gymnasium.spaces import Box
from omni.isaac.core.articulations import ArticulationView
from torch import Tensor


def dict_to_box_constraints(joint_constraints: JointsConstraints) -> Box:
    joint_names = list(joint_constraints.keys())

    low_joint_constraints = np.zeros(
        (len(joint_names)),
        dtype=np.float32,
    )
    high_joint_constraints = np.zeros_like(low_joint_constraints)

    # Ensures that the joint constraints are in the correct order
    for i, joint_name in enumerate(joint_names):
        constraint = joint_constraints[joint_name]

        low_joint_constraints[i] = constraint[0]
        high_joint_constraints[i] = constraint[1]

    box_joint_constraints = Box(
        low=low_joint_constraints,
        high=high_joint_constraints,
        dtype=np.float32,
    )

    return box_joint_constraints


def apply_joint_constraints(
    box_joint_constraints: Box,
    prim_paths: List[str],
    joint_names: List[str],
) -> None:
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdPhysics

    stage = get_current_stage()

    for i, prim_path in enumerate(prim_paths):
        for j, joint_name in enumerate(joint_names):
            joint_path = f"{prim_path}/{joint_name}"

            rev_joint = UsdPhysics.RevoluteJoint.Get(stage, joint_path)

            if not rev_joint:
                continue  # Skip if joint is not a RevoluteJoint

            rev_joint.CreateLowerLimitAttr().Set(box_joint_constraints.low[j].item())
            rev_joint.GetUpperLimitAttr().Set(box_joint_constraints.high[j].item())


class VecJointsController:
    def __init__(
        self,
        universe: Universe,
        noise_function: NoiseFunction,
        joint_constraints: JointsConstraints,
    ):
        self.path_expr: str = ""

        self._universe: Universe = universe
        self._articulation_view: Optional[ArticulationView] = None

        self._noise_function: NoiseFunction = noise_function
        self._target_joint_positions: Tensor = torch.zeros(0)
        self.joint_constraints: JointsConstraints = joint_constraints
        self.box_joint_constraints: Box = dict_to_box_constraints(joint_constraints)

        self._is_constructed: bool = False

    @property
    def art_view(self) -> Optional[ArticulationView]:
        if not self._is_constructed:
            return None

        return self._articulation_view

    @property
    def target_joint_positions(self) -> Tensor:
        return self._target_joint_positions

    def construct(self, path_expr: str) -> None:
        assert (
            not self._is_constructed
        ), "Joints controller already constructed: tried to construct!"

        self.path_expr = path_expr

        from omni.isaac.core.articulations import ArticulationView

        self._articulation_view = ArticulationView(
            self.path_expr,
            name="joints_controller_art_view",
            reset_xform_properties=False,
        )
        self._universe.add_prim(self._articulation_view)

        self._universe.reset()

        apply_joint_constraints(
            self.box_joint_constraints,
            self._articulation_view.prim_paths,
            self._articulation_view.joint_names,
        )

        self._is_constructed = True

    def step(self, joint_actions: Tensor) -> None:
        assert self._is_constructed, "Joints controller not constructed: tried to step!"

        self._target_joint_positions = self._process_joint_actions(
            joint_actions,
            self.box_joint_constraints,
            self._noise_function,
        )

        self._articulation_view.set_joint_position_targets(self._target_joint_positions)

    def reset(
        self,
        joint_positions: Tensor,
        joint_velocities: Tensor,
        joint_efforts: Tensor,
        indices: Optional[Indices] = None,
    ) -> None:
        assert (
            self._is_constructed
        ), "Joints controller not constructed: tried to reset!"

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
            self.box_joint_constraints,
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
            torch.from_numpy(self.box_joint_constraints.low).to(
                joint_positions.device,
            ),
            torch.from_numpy(self.box_joint_constraints.high).to(
                joint_positions.device,
            ),
            -1.0,
            1.0,
        )

        return joint_positions_normalized

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

        # TODO: Normalize joint velocities according to their actual limits
        #   As opposed to using the joints' constraints

        return self.normalize_joint_positions(self.get_joint_velocities_deg())

    def get_joint_positions_deg(self) -> Tensor:
        return torch.rad2deg(self._articulation_view.get_joint_positions())

    def get_joint_velocities_deg(self) -> Tensor:
        return torch.rad2deg(self._articulation_view.get_joint_velocities())

    def _process_joint_actions(
        self,
        joint_actions: Tensor,
        box_joint_constraints: Box,
        noise_function: Optional[NoiseFunction] = None,
    ) -> Tensor:
        """
        Joint actions are processed by mapping them to the joint constraints (degrees) and applying noise.
        Args:
            joint_actions: The joint actions to be processed [-1, 1].
            box_joint_constraints: The joint constraints, as a gymnasium.Box object.
            noise_function: The noise function to be applied to the computed joint positions.

        Returns:
            The processed joint positions (in radians).
        """
        low_constraints_t = torch.from_numpy(box_joint_constraints.low).to(
            device=self._universe.device
        )
        high_constraints_t = torch.from_numpy(box_joint_constraints.high).to(
            device=self._universe.device
        )

        joint_positions = torch.zeros_like(joint_actions).to(self._universe.device)

        for i, _ in enumerate(joint_positions):
            joint_positions[i] = torch.clamp(
                joint_actions[i],
                min=-1.0,
                max=1.0,
            )

            joint_positions[i] = torch.lerp(
                low_constraints_t,
                high_constraints_t,
                (joint_positions[i] + 1) / 2,
            )

            if noise_function is not None:
                joint_positions[i] = noise_function(joint_positions[i])

            joint_positions[i] = torch.deg2rad(joint_positions[i])

        return joint_positions
