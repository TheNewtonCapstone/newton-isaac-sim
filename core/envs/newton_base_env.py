from abc import abstractmethod
from typing import Optional

import torch as th

from ..agents import NewtonBaseAgent
from ..archiver import Archiver
from ..domain_randomizer import NewtonBaseDomainRandomizer
from .base_env import BaseEnv
from ..logger import Logger
from ..terrain.terrain import Terrain
from ..types import EnvObservations, Actions, Indices
from ..universe import Universe


class NewtonBaseEnv(BaseEnv):
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
        )

        self.agent: NewtonBaseAgent = agent
        self.domain_randomizer: NewtonBaseDomainRandomizer = domain_randomizer
        self._sub_terrain_origins: Optional[th.Tensor] = None

        from core.utils.math import IDENTITY_QUAT

        self.reset_newton_positions: th.Tensor = th.zeros((self.num_envs, 3))
        self.reset_newton_orientations: th.Tensor = IDENTITY_QUAT.repeat(
            self.num_envs, 1
        )

        self._inverse_control_frequency = inverse_control_frequency

    def pre_build(self) -> None:
        super().pre_build()

        self.terrain.register_self()
        self.agent.register_self()

        self.domain_randomizer.register_self()
        self.domain_randomizer.set_initial_orientations(self.reset_newton_orientations)

        Logger.info(
            f"{self.__class__.__name__} pre-built with {self.num_envs} environments"
        )

        self._is_pre_built = True

    def post_build(self):
        super().post_build()

        self._sub_terrain_origins = th.from_numpy(self.terrain.sub_terrain_origins).to(
            self._universe.device,
            dtype=th.float32,
        )

        # Convert to the correct device
        self.reset_newton_positions = self._compute_agent_reset_positions(
            th.ones((self.num_envs,)) * 0.0
        )

        self.domain_randomizer.set_initial_positions(self.reset_newton_positions)

        Logger.info(
            f"{self.__class__.__name__} post-built and generated starting positions"
        )

        self._is_post_built = True

    @abstractmethod
    def step(self, actions: Actions, render: bool = True) -> None:
        # in some cases, we want the simulation to have a higher resolution than the agent's control frequency
        for i in range(self._inverse_control_frequency):
            self.agent.step(actions)  # agent runs physic-related computations

            self.domain_randomizer.on_step()  # DR should always happen before any physics step

            super().step(actions, i == 0)  # advances the simulation by one step

    @abstractmethod
    def reset(self, indices: Optional[Indices] = None) -> EnvObservations:
        self.domain_randomizer.on_reset(
            indices
        )  # DR should always happen before any physics reset

        super().reset(indices)

        return self.get_observations()

    @abstractmethod
    def get_observations(self) -> EnvObservations:
        env_obs = self.agent.get_observations()

        env_obs["world_gravities"] = (
            th.tensor(
                [0.0, 0.0, self._universe.gravity],
                device=self._universe.device,
            )
        ).repeat(self.num_envs, 1)

        Archiver.put(
            "env_obs",
            {"world_gravity": th.tensor([0.0, 0.0, self._universe.gravity])},
        )

        return env_obs

    def _compute_agent_reset_positions(self, agent_heights: th.Tensor) -> th.Tensor:
        # Flatten terrain origins
        flat_origins = self._sub_terrain_origins

        # Spawn the agents in the first subterrain if curriculum
        if self.terrain.curriculum:
            agent_origins_indices = th.zeros_like(agent_heights, dtype=th.int32)
        else:
            # Randomly sample terrain origins for each agent
            agent_origins_indices = th.randint(
                0, flat_origins.shape[0], (self.num_envs,)
            )
        agent_origins = flat_origins[agent_origins_indices]

        # Add agent heights to account for the varying agent sizes
        agent_heights = agent_heights.to(device=agent_origins.device)
        agent_origins[:, 2] += agent_heights

        return agent_origins
