from typing import Callable, Optional, List

from gymnasium.spaces import Box
from torch import Tensor

import torch
from core.types import NoiseFunction
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView


class VecJointsController:
    def __init__(
        self,
        path_expr: str,
        world: World,
        noise_function: Callable[[Tensor], Tensor],
        joint_constraints: Box,
    ):
        self.path_expr: str = path_expr

        self.world: World = world
        self._articulation_view: Optional[ArticulationView] = None

        self._noise_function: NoiseFunction = noise_function
        self._target_joint_positions: Tensor = torch.zeros(0)
        self._joint_constraints: Box = joint_constraints

        self._is_constructed: bool = False

        # TODO: add constraints and a per-drive class

    @property
    def art_view(self):
        if not self._is_constructed:
            return 0

        return self._articulation_view

    @property
    def target_joint_positions(self) -> Tensor:
        return self._target_joint_positions

    def construct(self) -> None:
        if self._is_constructed:
            return

        from omni.isaac.core.articulations import ArticulationView

        self._articulation_view = ArticulationView(
            self.path_expr,
            name="joints_controller_art_view",
        )
        self.world.scene.add(self._articulation_view)

        self.world.reset()

        assert self._joint_constraints.shape[0] == len(
            self._articulation_view.joint_names
        ), "Joint constraints must match the number of joints"

        self._apply_joint_constraints(
            self._joint_constraints,
            self._articulation_view.prim_paths,
            self._articulation_view.joint_names,
        )

        self._is_constructed = True

    def step(self, joint_positions: Tensor) -> None:
        assert self._is_constructed, "Joints controller must be constructed first"

        self._target_joint_positions = self._process_joint_positions(
            joint_positions,
            self._joint_constraints,
            self._noise_function,
        )

        # TODO
        self._articulation_view.set_joint_position_targets(self._target_joint_positions)

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
        noise_function: NoiseFunction,
    ) -> Tensor:
        joint_positions = noise_function(joint_positions)

        for i, _ in enumerate(joint_positions):
            joint_positions[i] = torch.clamp(
                joint_positions[i],
                torch.from_numpy(joint_constraints.low).to(device=self.world.device),
                torch.from_numpy(joint_constraints.high).to(device=self.world.device),
            )

        right_side_joint_indices = self._get_right_side_shoulder_indices()
        for i in right_side_joint_indices:
            joint_positions[i] = -joint_positions[i]

        return joint_positions

    def _get_right_side_shoulder_indices(self) -> List[int]:
        return [
            i
            for i, name in enumerate(self._articulation_view.joint_names)
            if "R_HAA" in name
        ]
