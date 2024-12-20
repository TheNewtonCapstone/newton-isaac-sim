from typing import Optional, List

import torch
from core.types import NoiseFunction, Indices
from core.universe import Universe
from gymnasium.spaces import Box
from omni.isaac.core.articulations import ArticulationView
from torch import Tensor


class VecJointsController:
    def __init__(
        self,
        universe: Universe,
        noise_function: NoiseFunction,
        joint_constraints: Box,
    ):
        self.path_expr: str = ""

        self.universe: Universe = universe
        self._articulation_view: Optional[ArticulationView] = None

        self._noise_function: NoiseFunction = noise_function
        self._target_joint_positions: Tensor = torch.zeros(0)
        self.joint_constraints: Box = joint_constraints

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
        )
        self.universe.add_to_scene(self._articulation_view)

        self.universe.reset()

        assert self.joint_constraints.shape[0] == len(
            self._articulation_view.joint_names
        ), "Joint constraints must match the number of joints"

        self._apply_joint_constraints(
            self.joint_constraints,
            self._articulation_view.prim_paths,
            self._articulation_view.joint_names,
        )

        self._is_constructed = True

    def step(self, joint_positions: Tensor) -> None:
        assert self._is_constructed, "Joints controller not constructed: tried to step!"

        self._target_joint_positions = self._process_joint_positions(
            joint_positions,
            self.joint_constraints,
            self._noise_function,
        )

        # TODO
        self._articulation_view.set_joint_position_targets(self._target_joint_positions)

    def reset(self, joint_positions: Tensor, indices: Indices = None) -> None:
        assert (
            self._is_constructed
        ), "Joints controller not constructed: tried to reset!"

        if indices is None:
            indices = torch.arange(
                self._articulation_view.count,
                device=self.universe.physics_device,
            )
        else:
            indices = torch.from_numpy(indices).to(device=self.universe.physics_device)

        joint_positions = joint_positions.to(device=self.universe.physics_device)

        self._target_joint_positions = self._process_joint_positions(
            joint_positions,
            self.joint_constraints,
        )

        self._articulation_view.set_joint_position_targets(
            self._target_joint_positions,
            indices,
        )

    def _apply_joint_constraints(
        self,
        joint_constraints: Box,
        prim_paths: List[str],
        joint_names: List[str],
    ) -> None:
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import UsdPhysics

        stage = get_current_stage()

        for i, prim_path in enumerate(prim_paths):
            for j, joint_name in enumerate(joint_names):
                joint_path = f"{prim_path}/{joint_name}"

                joint_limits = UsdPhysics.RevoluteJoint.Get(stage, joint_path)
                joint_limits.CreateLowerLimitAttr().Set(joint_constraints.low[j].item())
                joint_limits.GetUpperLimitAttr().Set(joint_constraints.high[j].item())

    def _process_joint_positions(
        self,
        joint_positions: Tensor,
        joint_constraints: Box,
        noise_function: Optional[NoiseFunction] = None,
    ) -> Tensor:
        if noise_function is not None:
            joint_positions = noise_function(joint_positions)

        for i, _ in enumerate(joint_positions):
            joint_positions[i] = torch.clamp(
                joint_positions[i],
                torch.from_numpy(joint_constraints.low).to(
                    device=self.universe.physics_device
                ),
                torch.from_numpy(joint_constraints.high).to(
                    device=self.universe.physics_device
                ),
            )

        right_side_joint_indices = self._get_right_side_shoulder_indices()
        for i in right_side_joint_indices:
            joint_positions[:, i] = -joint_positions[:, i]

        return joint_positions

    def _get_right_side_shoulder_indices(self) -> List[int]:
        return [
            i
            for i, name in enumerate(self._articulation_view.joint_names)
            if "R_HAA" in name
        ]
