from abc import abstractmethod
from typing import List

import torch
from core.agents import NewtonBaseAgent
from core.domain_randomizer import NewtonBaseDomainRandomizer
from core.envs import BaseEnv
from core.terrain import BaseTerrainBuilder
from core.types import Observations, Actions, Indices
from core.universe import Universe
from torch import Tensor


class NewtonBaseEnv(BaseEnv):
    def __init__(
        self,
        agent: NewtonBaseAgent,
        num_envs: int,
        terrain_builders: List[BaseTerrainBuilder],
        domain_randomizer: NewtonBaseDomainRandomizer,
        inverse_control_frequency: int,
    ):
        super().__init__(
            agent,
            num_envs,
            terrain_builders,
            domain_randomizer,
        )

        self.agent: NewtonBaseAgent = agent
        self.domain_randomizer: NewtonBaseDomainRandomizer = domain_randomizer

        self.reset_newton_positions: Tensor = torch.zeros((self.num_envs, 3))
        self.reset_newton_rotations: Tensor = torch.tile(
            torch.tensor([0.0, 0.0, 0.0, 1.0]), (self.num_envs, 1)
        )

        self._inverse_control_frequency = inverse_control_frequency

    @abstractmethod
    def construct(self, universe: Universe) -> None:
        super().construct(universe)

    @abstractmethod
    def step(self, actions: Actions) -> Observations:
        # in some cases, we want the simulation to have a higher resolution than the agent's control frequency
        for _ in range(self._inverse_control_frequency):
            super().step(actions)  # advances the simulation by one step

        self.domain_randomizer.on_step()

        return self.get_observations()

    @abstractmethod
    def reset(self, indices: Indices = None) -> Observations:
        super().reset(indices)

        self.domain_randomizer.on_reset(indices)

        return self.get_observations()

    @abstractmethod
    def get_observations(self) -> Observations:
        return self.agent.get_observations()
