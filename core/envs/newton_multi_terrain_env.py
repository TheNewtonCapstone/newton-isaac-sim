from typing import List, Optional

import torch
from ..agents import NewtonBaseAgent
from ..domain_randomizer import NewtonBaseDomainRandomizer
from . import NewtonBaseEnv
from ..logger import Logger
from ..terrain import BaseTerrainBuilder
from ..types import EnvObservations, Actions, Indices
from ..universe import Universe


class NewtonMultiTerrainEnv(NewtonBaseEnv):
    def __init__(
        self,
        universe: Universe,
        agent: NewtonBaseAgent,
        num_envs: int,
        terrain_builders: List[BaseTerrainBuilder],
        domain_randomizer: NewtonBaseDomainRandomizer,
        inverse_control_frequency: int,
    ):
        super().__init__(
            universe,
            agent,
            num_envs,
            terrain_builders,
            domain_randomizer,
            inverse_control_frequency,
        )

    def construct(self) -> None:
        super().construct()

        num_terrains = len(self.terrain_builders)
        terrains_size = self.terrain_builders[0].size

        import math

        # generates a list of positions for each of the terrains, in a grid pattern
        num_terrains_side = math.ceil(math.sqrt(num_terrains))
        terrain_positions = torch.tensor(
            [
                [
                    (i % num_terrains_side) * terrains_size - terrains_size / 2,
                    (i // num_terrains_side) * terrains_size - terrains_size / 2,
                    0,
                ]
                for i in range(num_terrains)
            ]
        ).tolist()

        agent_batch_qty = int(math.ceil(self.num_envs / num_terrains))

        # build & add all given terrains
        for i, terrain_builder in enumerate(self.terrain_builders):
            terrain_spawn_position = terrain_positions[i]

            assert (
                terrain_builder.size == terrains_size
            ), "All terrains must have the same size"

            self.terrain_builds.append(
                terrain_builder.build_from_self(terrain_spawn_position)
            )

            # we want all agents to be evenly split across all terrains
            agent_batch_start = i * agent_batch_qty
            agent_batch_end = i * agent_batch_qty + agent_batch_qty

            self.reset_newton_positions[agent_batch_start:agent_batch_end, :] = (
                torch.tensor(
                    [
                        terrain_spawn_position[0],
                        terrain_spawn_position[1],
                        0.4 + self.terrain_builds[i].height,
                    ]
                )
            )

        # in some cases, ceil will give us more positions than we need
        if len(self.reset_newton_positions) > self.num_envs:
            self.reset_newton_positions = self.reset_newton_positions[: self.num_envs]

        self.agent.register_self()

        self.domain_randomizer.register_self()
        # TODO: investigate whether we need to have the positions and rotations in this class or in domain randomizer
        self.domain_randomizer.set_initial_positions(self.reset_newton_positions)
        self.domain_randomizer.set_initial_orientations(self.reset_newton_orientations)

        Logger.info(
            f"NewtonMultiTerrainEnv constructed with {num_terrains} terrains and {self.num_envs} agents."
        )

        self._is_constructed = True

    def post_construct(self):
        super().post_construct()

        Logger.info("NewtonMultiTerrainEnv post-constructed.")

        self._is_post_constructed = True

    def step(self, actions: Actions) -> None:
        super().step(actions)  # advances the simulation by one step

    def reset(self, indices: Optional[Indices] = None) -> EnvObservations:
        super().reset(indices)

        return self.get_observations()

    def get_observations(self) -> EnvObservations:
        return super().get_observations()
