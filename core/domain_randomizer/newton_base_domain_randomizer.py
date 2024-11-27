from typing import Optional, List

import torch

from core.agents import NewtonBaseAgent
from core.domain_randomizer.base_domain_randomizer import BaseDomainRandomizer
from core.terrain import BaseTerrainBuild
from core.types import Settings, Indices
from omni.isaac.core.articulations import ArticulationView


class NewtonBaseDomainRandomizer(BaseDomainRandomizer):
    def __init__(
        self,
        seed: int,
        agent: NewtonBaseAgent,
        terrain_builds: List[BaseTerrainBuild],
        randomizer_settings: Settings,
    ):
        super().__init__(
            seed,
            agent,
            terrain_builds,
            randomizer_settings,
        )

        self._agent: NewtonBaseAgent = agent
        self._terrain_builds: List[BaseTerrainBuild] = terrain_builds

        self._newton_art_view: Optional[ArticulationView] = None

    def construct(self) -> None:
        super().construct()

        self._newton_art_view = ArticulationView(
            prim_paths_expr=self._agent.base_path_expr,
            name="newton_dr_art_view",
        )
        self._env.world.scene.add(self._newton_art_view)

        self._is_constructed = True

    def on_step(self) -> None:
        super().on_step()

    def on_reset(self, indices: Indices = None) -> None:
        super().on_reset(indices)

        num_to_reset = indices.shape[0]

        # TODO: decide where the reset positions and rotations should come from
        self._newton_art_view.set_world_poses(
            positions=self._env.reset_newton_positions[indices],
            orientations=self._env.reset_newton_rotations[indices],
            indices=indices,
        )

        # using set_velocities instead of individual methods (lin & ang),
        # because it's the only method supported in the GPU pipeline
        self._newton_art_view.set_velocities(
            torch.zeros((num_to_reset, 6), dtype=torch.float32),
            indices,
        )

        self._newton_art_view.set_joint_efforts(
            torch.zeros((num_to_reset, 12), dtype=torch.float32),
            indices,
        )

        self._newton_art_view.set_joint_velocities(
            torch.zeros((num_to_reset, 12), dtype=torch.float32),
            indices,
        )

        # TODO: make this less convoluted by putting the joint constraints in the agent
        joint_positions = torch.tensor(
            [
                self._agent.joints_controller._joint_constraints.sample()
                for _ in range(num_to_reset)
            ]
        )

        self._newton_art_view.set_joint_positions(
            joint_positions,
            indices,
        )
