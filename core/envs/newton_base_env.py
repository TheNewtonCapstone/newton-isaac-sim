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

        return self.get_observations()

    @abstractmethod
    def get_observations(self) -> Observations:
        return self.agent.get_observations()
