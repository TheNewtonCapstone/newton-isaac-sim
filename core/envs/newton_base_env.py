from abc import abstractmethod
from typing import Optional

import torch as th

from ..agents import NewtonBaseAgent
from ..archiver import Archiver
from ..domain_randomizer import NewtonBaseDomainRandomizer
from .base_env import BaseEnv
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

        Logger.info(f"NewtonTerrainEnv constructed with {self.num_envs} environments")

        self._is_constructed = True

    def post_build(self):
        super().post_build()

        self._sub_terrain_origins = th.from_numpy(self.terrain.sub_terrain_origins).to(
            self._universe.device,
            dtype=th.float32,
        )

        # Convert to the correct device
        self.reset_newton_positions = self._compute_agent_reset_positions(
            th.ones((self.num_envs,)) * self.agent.transformed_position[2]
        )

        self.domain_randomizer.set_initial_positions(self.reset_newton_positions)

        Logger.info(
            "NewtonTerrainEnv post-constructed and generated starting positions"
        )

        self._is_post_constructed = True

    @abstractmethod
    def step(self, actions: Actions) -> None:
        # in some cases, we want the simulation to have a higher resolution than the agent's control frequency
        for _ in range(self._inverse_control_frequency):
            self.agent.step(actions)  # agent runs physic-related computations

            self.domain_randomizer.on_step()  # DR should always happen before any physics step

            super().step(actions)  # advances the simulation by one step

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

        gravity_direction, gravity_magnitude = (
            self._universe.get_physics_context().get_gravity()
        )

        env_obs["world_gravities"] = (
            th.tensor(
                gravity_direction,
                device=self._universe.device,
            )
        ).repeat(self.num_envs, 1)

        Archiver.put(
            "env_obs",
            {"world_gravity": th.tensor(gravity_direction) * gravity_magnitude},
        )

        return env_obs
