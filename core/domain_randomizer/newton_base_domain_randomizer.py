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
        self.initial_rotations: torch.Tensor = torch.zeros((1, 4))

    def construct(self, universe: Universe) -> None:
        super().construct(universe)

        self._newton_art_view = ArticulationView(
            prim_paths_expr=self._agent.base_path_expr,
            name="newton_dr_art_view",
        )
        self._universe.add_to_scene(self._newton_art_view)

        self._is_constructed = True

    def on_step(self) -> None:
        super().on_step()

    def on_reset(self, indices: Indices = None) -> None:
        super().on_reset(indices)

        if indices is None:
            indices = torch.arange(self._agent.num_agents)
        else:
            indices = torch.from_numpy(indices).to(device=self._universe.physics_device)

        num_to_reset = indices.shape[0]

        # TODO: decide where the reset positions and rotations should come from
        self._newton_art_view.set_world_poses(
            positions=self.initial_positions[indices],
            orientations=self.initial_rotations[indices],
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

        import numpy as np

        # TODO: make this less convoluted by putting the joint constraints in the agent
        joint_positions = torch.from_numpy(
            np.array(
                [
                    self._agent.joints_controller._joint_constraints.sample()
                    for _ in range(num_to_reset)
                ]
            )
        )

        self._newton_art_view.set_joint_positions(
            joint_positions,
            indices,
        )

    def set_initial_positions(self, positions: torch.Tensor) -> None:
        self.initial_positions = positions.to(self._universe.physics_device)

    def set_initial_rotations(self, rotations: torch.Tensor) -> None:
        self.initial_rotations = rotations.to(self._universe.physics_device)
