from typing import Optional

import torch as th

from . import NewtonBaseEnv
from ..agents import NewtonBaseAgent
from ..domain_randomizer import NewtonBaseDomainRandomizer
from ..logger import Logger
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

        self.agent.register_self()

        self.domain_randomizer.register_self()
        self.domain_randomizer.set_initial_orientations(self.reset_newton_orientations)

        Logger.info(f"NewtonTerrainEnv constructed with {self.num_envs} environments")

        self._is_constructed = True

    def post_construct(self):
        super().post_construct()

        env_origins = th.from_numpy(
            self.terrain.sub_terrain_origins
        )  # Shape (num_rows, num_cols, 3)
        env_origins[
            :, :, 2
        ] += 0.3  # Add a small offset to the z-axis to accommodate the agent's height

        # Flatten terrain origins to a (num_envs, 3) shape
        flat_origins = env_origins.reshape(-1, 3)

        # Randomly sample terrain origins for each agent
        agent_origins_indices = th.randint(0, flat_origins.shape[0], (self.num_envs,))
        agent_origins = flat_origins[agent_origins_indices]

        # Convert to the correct device
        self.reset_newton_positions = agent_origins.to(
            self._universe.device,
            dtype=th.float32,
        )

        self.domain_randomizer.set_initial_positions(self.reset_newton_positions)

        Logger.info(
            f"NewtonTerrainEnv post-constructed and generated starting positions"
        )

        self._is_post_constructed = True

    def step(self, actions: Actions) -> None:
        super().step(actions)  # advances the simulation by one step

    def reset(self, indices: Optional[Indices] = None) -> EnvObservations:
        super().reset(indices)

        return self.get_observations()

    def get_observations(self) -> EnvObservations:
        return super().get_observations()
