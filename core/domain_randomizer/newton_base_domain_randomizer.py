from typing import Optional

import torch
from genesis.engine.entities import RigidEntity

from .base_domain_randomizer import BaseDomainRandomizer
from ..agents import NewtonBaseAgent
from ..types import Config, Indices
from ..universe import Universe


class NewtonBaseDomainRandomizer(BaseDomainRandomizer):
    def __init__(
        self,
        universe: Universe,
        seed: int,
        agent: NewtonBaseAgent,
        randomizer_settings: Config,
    ):
        super().__init__(
            universe,
            seed,
            agent,
            randomizer_settings,
        )

        self._agent: NewtonBaseAgent = agent

        self._robot: Optional[RigidEntity] = None
        self.initial_positions: torch.Tensor = torch.zeros(
            (1, 3), device=self._universe.device
        )
        self.initial_orientations: torch.Tensor = torch.zeros(
            (1, 4), device=self._universe.device
        )

    def pre_build(self) -> None:
        super().pre_build()

        self._is_pre_built = True

    def post_build(self) -> None:
        super().post_build()

        self._is_post_built = True

    def on_step(self) -> None:
        super().on_step()

    def on_reset(self, indices: Indices = None) -> None:
        super().on_reset(indices)

        if indices is None:
            indices = torch.arange(self._agent.num_agents)
        else:
            indices = indices.to(device=self._universe.device)

        num_to_reset = indices.shape[0]

        self._agent.robot.set_pos(
            pos=self.initial_positions[indices],
            envs_idx=indices,
            zero_velocity=True,
        )

        self._agent.robot.set_quat(
            quat=self.initial_orientations[indices],
            envs_idx=indices,
            zero_velocity=True,
        )

        # using set_velocities instead of individual methods (lin & ang),
        # because it's the only method supported in the GPU pipeline (default pipeline)
        joint_positions = torch.zeros((num_to_reset, 12), dtype=torch.float32)
        # (
        #    torch.rand((num_to_reset, 12), dtype=torch.float32) * 2.0 - 1.0
        # )  # [-1, 1]

        joint_velocities = torch.zeros_like(joint_positions)
        joint_efforts = torch.zeros_like(joint_positions)

        self._agent.joints_controller.reset(
            joint_positions,
            joint_velocities,
            joint_efforts,
            indices,
        )

    def set_initial_positions(self, positions: torch.Tensor) -> None:
        self.initial_positions = positions.to(self._universe.device)

    def set_initial_orientations(self, orientations: torch.Tensor) -> None:
        self.initial_orientations = orientations.to(self._universe.device)

    def set_initial_position(
        self, indices: torch.Tensor, positions: torch.Tensor
    ) -> None:
        self.initial_positions[indices] = positions.to(self._universe.device)
