from typing import Optional

import torch
from core.agents import NewtonBaseAgent
from core.domain_randomizer.base_domain_randomizer import BaseDomainRandomizer
from core.types import Settings, Indices
from core.universe import Universe
from omni.isaac.core.prims import RigidPrimView


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

        self._rigid_prim_view: Optional[RigidPrimView] = None
        self.initial_positions: torch.Tensor = torch.zeros((1, 3))
        self.initial_orientations: torch.Tensor = torch.zeros((1, 4))

    def construct(self, universe: Universe) -> None:
        super().construct(universe)

        self._rigid_prim_view = RigidPrimView(
            prim_paths_expr=self._agent.base_path_expr,
            name="newton_dr_art_view",
            reset_xform_properties=False,
        )
        self._universe.add_prim(self._rigid_prim_view)

        self._universe.reset()

        self._is_constructed = True

    def on_step(self) -> None:
        super().on_step()

    def on_reset(self, indices: Indices = None) -> None:
        super().on_reset(indices)

        if indices is None:
            indices = torch.arange(self._agent.num_agents)
        else:
            indices = indices.to(device=self._universe.device)

        num_to_reset = indices.shape[0]

        # TODO: decide where the reset positions and rotations should come from
        self._rigid_prim_view.set_world_poses(
            positions=self.initial_positions[indices],
            orientations=self.initial_orientations[indices],
            indices=indices,
            usd=self._universe.use_usd_physics,
        )

        # using set_velocities instead of individual methods (lin & ang),
        # because it's the only method supported in the GPU pipeline (default pipeline)
        self._rigid_prim_view.set_velocities(
            torch.zeros((num_to_reset, 6), dtype=torch.float32),
            indices,
        )

        joint_positions = (
            torch.rand((num_to_reset, 12), dtype=torch.float32) * 2.0 - 1.0
        )  # [-1, 1]

        joint_velocities = torch.zeros_like(joint_positions)
        joint_efforts = torch.zeros_like(joint_positions)

        self._agent.joints_controller.reset(
            joint_positions,
            joint_velocities,
            joint_efforts,
            indices,
        )

    def set_initial_positions(self, positions: torch.Tensor) -> None:
        self.initial_positions = positions.to(self._universe.device)

    def set_initial_orientations(self, orientations: torch.Tensor) -> None:
        self.initial_orientations = orientations.to(self._universe.device)
