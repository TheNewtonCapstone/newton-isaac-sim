from typing import List

import torch
from core.agents import NewtonBaseAgent
from core.domain_randomizer import NewtonBaseDomainRandomizer
from core.envs import NewtonBaseEnv
from core.terrain import BaseTerrainBuilder
from core.types import Observations, Actions, Indices
from core.universe import Universe


class NewtonMultiTerrainEnv(NewtonBaseEnv):
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
            inverse_control_frequency,
        )

    def construct(self, universe: Universe) -> None:
        super().construct(universe)

        num_terrains = len(self.terrain_builders)
        terrains_size = self.terrain_builders[0].size

        import math

        # generates a list of positions for each of the terrains, in a grid pattern
        num_terrains_side = math.ceil(math.sqrt(num_terrains))
        terrain_positions = torch.tensor(
            [
                [
                    (i % num_terrains_side) * terrains_size[0] - terrains_size[0] / 2,
                    (i // num_terrains_side) * terrains_size[1] - terrains_size[1] / 2,
                    0,
                ]
                for i in range(num_terrains)
            ]
        ).tolist()

        agent_batch_qty = int(math.ceil(self.num_envs / num_terrains))

        from core.utils.physics import raycast

        # build & add all given terrains
        for i, terrain_builder in enumerate(self.terrain_builders):
            terrain_spawn_position = terrain_positions[i]

            assert terrain_builder.size.equal(
                terrains_size
            ), "All terrains must have the same size"

            self.terrain_builds.append(
                terrain_builder.build_from_self(terrain_spawn_position)
            )

        # propagate physics changes
        self.universe.reset()

        for i, _ in enumerate(self.terrain_builds):
            terrain_spawn_position = terrain_positions[i]

            # from the raycast, we can get the desired position of the agent to avoid clipping with the terrain
            raycast_height = 25
            max_ray_test_dist = 100
            min_ray_dist = max_ray_test_dist
            num_rays = 9
            rays_side = math.isqrt(num_rays)
            ray_separation = 0.15

            for j in range(num_rays):
                # we also want to cover a grid of rays on the xy-plane
                start_x = -ray_separation * (rays_side / 2)
                start_y = -ray_separation * (rays_side / 2)
                ray_x = ray_separation * (j % rays_side) + start_x
                ray_y = ray_separation * (j // rays_side) + start_y

                _, _, dist = raycast(
                    [
                        terrain_spawn_position[0] + ray_x,
                        terrain_spawn_position[1] + ray_y,
                        raycast_height,
                    ],
                    [0, 0, -1],
                    max_distance=max_ray_test_dist,
                )

                min_ray_dist = min(dist, min_ray_dist)

            # we want all agents to be evenly split across all terrains
            agent_batch_start = i * agent_batch_qty
            agent_batch_end = i * agent_batch_qty + agent_batch_qty

            self.reset_newton_positions[agent_batch_start:agent_batch_end, :] = (
                torch.tensor(
                    [
                        terrain_spawn_position[0],
                        terrain_spawn_position[1],
                        0.4
                        + self.terrain_builds[
                            i
                        ].height,  # TODO: make this a better computed value
                    ]
                )
            )

        # in some cases, ceil will give us more positions than we need
        if len(self.reset_newton_positions) > self.num_envs:
            self.reset_newton_positions = self.reset_newton_positions[: self.num_envs]

        self.agent.construct(self.universe)

        self.domain_randomizer.construct(self.universe)
        # TODO: investigate whether we need to have the positions and rotations in this class or in domain randomizer
        self.domain_randomizer.set_initial_positions(self.reset_newton_positions)
        self.domain_randomizer.set_initial_orientations(self.reset_newton_orientations)

        self._is_constructed = True

        self.reset()

    def step(self, actions: Actions) -> Observations:
        self.agent.step(actions)  # has to be before the simulation advances

        super().step(actions)  # advances the simulation by one step

        return self.get_observations()

    def reset(self, indices: Indices = None) -> Observations:
        super().reset(indices)

        return self.get_observations()

    def get_observations(self) -> Observations:
        return super().get_observations()
