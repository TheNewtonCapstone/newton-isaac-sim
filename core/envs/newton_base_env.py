from abc import abstractmethod
from typing import List

import torch
from core.agents import NewtonBaseAgent
from core.envs import BaseEnv
from core.terrain import TerrainBuilder
from core.types import Observations, Settings


class NewtonBaseEnv(BaseEnv):
    def __init__(
        self,
        agent: NewtonBaseAgent,
        num_envs: int,
        terrain_builders: List[TerrainBuilder],
        world_settings: Settings,
        randomizer_settings: Settings,
    ):
        self.agent: NewtonBaseAgent

        super().__init__(
            agent,
            num_envs,
            terrain_builders,
            world_settings,
            randomizer_settings,
        )

    @abstractmethod
    def construct(self) -> None:
        super().construct()

        agent_root_path = f"{self.path}/agents"
        self.agent.construct(agent_root_path)

    @abstractmethod
    def step(self, actions: torch.Tensor, render: bool) -> Observations:
        super().step(actions, render)  # advances the simulation by one step

        return self.get_observations()

    @abstractmethod
    def reset(self) -> Observations:
        super().reset()

        return self.get_observations()

    @abstractmethod
    def get_observations(self) -> Observations:
        pass
