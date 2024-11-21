from typing import Callable, Optional

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
    ):
        self.path_expr: str = path_expr

        self.world: World = world
        self.articulation_view: Optional[ArticulationView] = None

        self._noise_function: NoiseFunction = noise_function
        self._target_joint_positions: Tensor = torch.zeros(0)
        self._is_constructed: bool = False

        # TODO: add constraints and a per-drive class

    @property
    def art_view(self):
        if not self._is_constructed:
            return 0

        return self.articulation_view

    @property
    def target_joint_positions(self) -> Tensor:
        return self._target_joint_positions

    def construct(self) -> None:
        if self._is_constructed:
            return

        from omni.isaac.core.articulations import ArticulationView

        self.articulation_view = ArticulationView(
            self.path_expr,
            name="joints_controller_art_view",
        )
        self.world.scene.add(self.articulation_view)

        self._is_constructed = True

    def step(self, joint_positions: Tensor) -> None:
        self._target_joint_positions = self._noise_function(joint_positions)

        # TODO
        current_joint_positions = self.articulation_view.get_joint_positions()
        target_joint_positions = self._target_joint_positions

        joint_positions_velocities = (
            current_joint_positions - target_joint_positions
        ) / self.world.get_physics_dt()
        joint_positions_velocities = joint_positions_velocities.clamp(-2, 2)

        self.articulation_view.set_joint_velocities(joint_positions_velocities)
