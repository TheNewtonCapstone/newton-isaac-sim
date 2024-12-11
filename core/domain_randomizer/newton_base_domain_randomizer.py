from typing import Optional, TYPE_CHECKING

import torch
from core.agents import NewtonBaseAgent
from core.domain_randomizer.base_domain_randomizer import BaseDomainRandomizer
from core.types import Settings, Indices
from core.universe import Universe
from omni.isaac.core.articulations import ArticulationView


class NewtonBaseDomainRandomizer(BaseDomainRandomizer):
    def __init__(
        self,
        seed: int,
        agent: NewtonBaseAgent,
        randomizer_settings: Settings,
    ):
        super().__init__(
            seed,
            agent,
            randomizer_settings,
        )

        self._agent: NewtonBaseAgent = agent

        self._newton_art_view: Optional[ArticulationView] = None
        self.initial_positions: torch.Tensor = torch.zeros((1, 3))
        self.initial_orientations: torch.Tensor = torch.zeros((1, 4))

    def construct(self, universe: Universe) -> None:
        super().construct(universe)

        self._newton_art_view = ArticulationView(
            prim_paths_expr=self._agent.base_path_expr,
            name="newton_dr_art_view",
        )
        self._universe.add_to_scene(self._newton_art_view)

        self._universe.reset()

        self._is_constructed = True

    def on_step(self) -> None:
        super().on_step()

    def on_reset(self, indices: Indices = None) -> None:
        super().on_reset(indices)

        indices_n = indices

        if indices is None:
            indices_t = torch.arange(self._agent.num_agents)
        else:
            indices_t = torch.from_numpy(indices).to(
                device=self._universe.physics_device
            )

        num_to_reset = indices_t.shape[0]

        # TODO: decide where the reset positions and rotations should come from
        self._newton_art_view.set_world_poses(
            positions=self.initial_positions[indices],
            orientations=self.initial_orientations[indices],
            indices=indices_t,
        )

        # using set_velocities instead of individual methods (lin & ang),
        # because it's the only method supported in the GPU pipeline (default pipeline)
        self._newton_art_view.set_velocities(
            torch.zeros((num_to_reset, 6), dtype=torch.float32),
            indices_t,
        )

        self._newton_art_view.set_joint_efforts(
            torch.zeros((num_to_reset, 12), dtype=torch.float32),
            indices_t,
        )

        self._newton_art_view.set_joint_velocities(
            torch.zeros((num_to_reset, 12), dtype=torch.float32),
            indices_t,
        )

        joint_positions = (
            torch.rand((num_to_reset, 12), dtype=torch.float32) * 2.0 - 1.0
        )  # [-1, 1]

        self._agent.joints_controller.reset(
            joint_positions,
            indices_n,
        )

    def set_initial_positions(self, positions: torch.Tensor) -> None:
        self.initial_positions = positions.to(self._universe.physics_device)

    def set_initial_orientations(self, orientations: torch.Tensor) -> None:
        self.initial_orientations = orientations.to(self._universe.physics_device)
