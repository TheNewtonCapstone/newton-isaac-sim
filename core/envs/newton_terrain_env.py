from typing import List, Optional

from gymnasium.experimental.wrappers.numpy_to_torch import numpy_to_torch

import torch
from ..agents import NewtonBaseAgent
from ..domain_randomizer import NewtonBaseDomainRandomizer
from . import NewtonBaseEnv
from ..terrain.terrain import Terrain
from ..types import EnvObservations, Actions, Indices
from ..universe import Universe


class NewtonTerrainEnv(NewtonBaseEnv):
    def __init__(
        self,
        universe: Universe,
        agent: NewtonBaseAgent,
        num_envs: int,
        terrain: Terrain,
        domain_randomizer: NewtonBaseDomainRandomizer,
        inverse_control_frequency: int,
    ):
        super().__init__(
            universe,
            agent,
            num_envs,
            terrain,
            domain_randomizer,
            inverse_control_frequency,
        )

    def construct(self) -> None:
        super().construct()

        # Change this to your actual number of agents
        env_origins = numpy_to_torch(self.terrain.env_origins)  # Shape (9, 8, 3)
        env_origins[:, :, 2] = (
            env_origins[:, :, 2] + 0.3
        )  # Add a small offset to the z-axis
        device = self._universe.device

        # Flatten terrain origins to a (num_envs, 3) shape
        flat_origins = env_origins.reshape(-1, 3)

        # Randomly sample terrain origins for each agent
        agent_origins_indices = torch.randint(
            0, flat_origins.shape[0], (self.num_envs,)
        )
        agent_origins = flat_origins[agent_origins_indices]

        # Convert to the correct device
        self.reset_newton_positions = agent_origins.to(device, dtype=torch.float32)

        print(
            f"Reset Newton positions: {self.reset_newton_positions} - {self.reset_newton_positions.shape}"
        )

        self.agent.register_self()

        self.domain_randomizer.register_self()
        # TODO: investigate whether we need to have the positions and rotations in this class or in domain randomizer
        self.domain_randomizer.set_initial_positions(self.reset_newton_positions)
        self.domain_randomizer.set_initial_orientations(self.reset_newton_orientations)

        self._is_constructed = True

    def post_construct(self):
        super().post_construct()

        self._is_post_constructed = True

    def step(self, actions: Actions) -> None:
        super().step(actions)  # advances the simulation by one step

    def reset(self, indices: Optional[Indices] = None) -> EnvObservations:
        super().reset(indices)

        return self.get_observations()

    def get_observations(self) -> EnvObservations:
        return super().get_observations()
