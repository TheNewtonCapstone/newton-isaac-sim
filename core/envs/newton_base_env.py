from abc import abstractmethod
from typing import List

import torch
from core.agents import NewtonBaseAgent
from core.envs import BaseEnv
from core.terrain import TerrainBuilder
from core.types import Observations, Settings, Actions, Indices
from torch import Tensor


class NewtonBaseEnv(BaseEnv):
    def __init__(
        self,
        agent: NewtonBaseAgent,
        num_envs: int,
        terrain_builders: List[TerrainBuilder],
        world_settings: Settings,
        randomizer_settings: Settings,
        inverse_control_frequency: int,
    ):
        super().__init__(
            agent,
            num_envs,
            terrain_builders,
            world_settings,
            randomizer_settings,
        )

        self.agent: NewtonBaseAgent = agent

        self.reset_newton_positions: Tensor = torch.zeros((self.num_envs, 3))
        self.reset_newton_rotations: Tensor = torch.tile(
            torch.tensor([0.0, 0.0, 0.0, 1.0]), (self.num_envs, 1)
        )

        self._inverse_control_frequency = inverse_control_frequency

    @abstractmethod
    def construct(self) -> None:
        super().construct()

    @abstractmethod
    def step(self, actions: Actions, render: bool) -> Observations:
        # in some cases, we want the simulation to have a higher resolution than the agent's control frequency
        for _ in range(self._inverse_control_frequency):
            super().step(actions, render)  # advances the simulation by one step

        return self.get_observations()

    @abstractmethod
    def reset(self, indices: Indices = None) -> Observations:
        super().reset(indices)

        if indices is None:
            self.world.reset()

            indices = torch.arange(self.num_envs)
        else:
            indices = torch.from_numpy(indices)

        num_to_reset = indices.shape[0]

        self.agent.newton_art_view.set_world_poses(
            positions=self.reset_newton_positions[indices],
            orientations=self.reset_newton_rotations[indices],
            indices=indices,
        )

        # using set_velocities instead of individual methods (lin & ang),
        # because it's the only method supported in the GPU pipeline
        self.agent.newton_art_view.set_velocities(
            torch.zeros((num_to_reset, 6), dtype=torch.float32),
            indices,
        )

        self.agent.newton_art_view.set_joint_efforts(
            torch.zeros((num_to_reset, 12), dtype=torch.float32),
            indices,
        )

        self.agent.newton_art_view.set_joint_velocities(
            torch.zeros((num_to_reset, 12), dtype=torch.float32),
            indices,
        )

        joint_positions = torch.tensor(
            [0.0, 0.0, 0.0] * 4,
            dtype=torch.float32,
        ).repeat(num_to_reset, 1)

        self.agent.newton_art_view.set_joint_positions(
            joint_positions,
            indices,
        )

        return self.get_observations()

    @abstractmethod
    def get_observations(self) -> Observations:
        return self.agent.get_observations()
