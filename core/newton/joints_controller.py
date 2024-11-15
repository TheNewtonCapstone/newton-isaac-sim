from typing import Callable, Optional

import torch
from core.types import NoiseFunction


class VecJointsController:
    def __init__(
        self,
        path_expr: str,
        noise_function: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.path_expr: str = path_expr

        from omni.isaac.core.articulations import ArticulationView

        self.articulation_view: Optional[ArticulationView] = None

        self._noise_function: NoiseFunction = noise_function
        self._target_joint_positions: torch.Tensor = torch.zeros(0)
        self._is_constructed: bool = False

    def __del__(self):
        del self.articulation_view

    @property
    def art_view(self):
        if not self._is_constructed:
            return 0

        return self.articulation_view

    @property
    def target_joint_positions(self) -> torch.Tensor:
        return self._target_joint_positions

    def construct(self) -> None:
        if self._is_constructed:
            return

        from omni.isaac.core.articulations import ArticulationView

        self.articulation_view = ArticulationView(self.path_expr)

        self._is_constructed = True

    def update(self, joint_positions: torch.Tensor) -> None:
        self._target_joint_positions = self._noise_function(joint_positions)

    def step(self) -> None:
        # TODO
        current_joint_positions = self.articulation_view.get_joint_positions()
        target_joint_positions = self._target_joint_positions

        delta_joint_positions = target_joint_positions - current_joint_positions
        delta_joint_positions = delta_joint_positions.clamp(-5.0, 5.0)

        self.articulation_view.set_joint_positions(
            current_joint_positions + delta_joint_positions
        )
