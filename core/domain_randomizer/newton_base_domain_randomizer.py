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
            (self.num_envs, 3),
            device=self.device,
        )
        self.initial_orientations: torch.Tensor = torch.zeros(
            (self.num_envs, 4),
            device=self.device,
        )

        self.initial_joint_positions: torch.Tensor = torch.zeros(
            (self.num_envs, 12),
            device=self.device,
        )  # radians

    def pre_build(self) -> None:
        super().pre_build()

        self._is_pre_built = True

    def post_build(self) -> None:
        super().post_build()

        self._is_post_built = True

    def on_step(self) -> None:
        super().on_step()

    def on_reset(self, indices: Indices = None) -> None:
        """
        Reset the domain randomizer and default state of the agent (pose and joints).
        Args:
            indices: Indices of the environments to reset. If None, reset all environments.

        Returns:
            None

        """
        super().on_reset(indices)

        if indices is None:
            indices = torch.arange(self.num_envs)
        else:
            indices = indices.to(device=self.device)

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

        self._agent.joints_controller.reset(
            joint_positions=self.initial_joint_positions[indices],
            indices=indices,
        )

    def set_initial_positions(
        self,
        positions: torch.Tensor,
        indices: Optional[Indices] = None,
    ) -> None:
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        else:
            indices = indices.to(self.device)

        self.initial_positions[indices] = positions.to(self.device)

    def set_initial_orientations(
        self,
        orientations: torch.Tensor,
        indices: Optional[Indices] = None,
    ) -> None:
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        else:
            indices = indices.to(self.device)

        self.initial_orientations[indices] = orientations.to(self.device)

    def set_initial_joint_positions(
        self,
        joint_positions: torch.Tensor,
        indices: Optional[Indices] = None,
    ) -> None:
        """
        Set the initial joint positions for the agent.
        Args:
            joint_positions: The joint positions (normalized) to set.
            indices: Indices of the environments to set the joint positions. If None, set all environments.

        Returns:
            None

        """
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        else:
            indices = indices.to(self.device)

        self.initial_joint_positions[indices] = joint_positions.to(self.device)
