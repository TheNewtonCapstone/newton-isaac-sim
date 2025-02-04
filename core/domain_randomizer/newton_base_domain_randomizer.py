from typing import Optional

import torch

from omni.isaac.core.prims import RigidPrimView, XFormPrim
from .base_domain_randomizer import BaseDomainRandomizer
from ..agents import NewtonBaseAgent
from ..types import Config, Indices
from ..universe import Universe
from ..terrain import Terrain
from ..utils.physics import set_physics_properties


class NewtonBaseDomainRandomizer(BaseDomainRandomizer):
    def __init__(
        self,
        universe: Universe,
        seed: int,
        agent: NewtonBaseAgent,
        randomizer_settings: Config,
        terrain: Terrain,
    ):
        super().__init__(
            universe,
            seed,
            agent,
            randomizer_settings,
            terrain,
        )

        self._agent: NewtonBaseAgent = agent
        self._rigid_prim_view: Optional[RigidPrimView] = None
        self._terrain_prim_view: Optional[XFormPrim] = None
        self.initial_positions: torch.Tensor = torch.zeros((1, 3))
        self.initial_orientations: torch.Tensor = torch.zeros((1, 4))
        self.friction_coefficients: torch.Tensor = torch.zeros((1, 3))
        self._frame_idx: int = 0

    def construct(self) -> None:
        super().construct()

        self._rigid_prim_view = RigidPrimView(
            prim_paths_expr=self._agent.base_path_expr,
            name="newton_dr_art_view",
            reset_xform_properties=False,
        )
        # self._terrain_prim_view = XFormPrim(
        #     prim_path=self.terrain.terrain_path,
        # )
        self._universe.add_prim(self._rigid_prim_view)

        self._is_constructed = True

    def post_construct(self) -> None:
        super().post_construct()

        self._is_post_constructed = True

    def on_step(self, indices: Indices = None) -> None:
        super().on_step()
        self._frame_idx += 1

        if indices is None:
            indices = torch.arange(self._agent.num_agents)
        else:
            indices = indices.to(device=self._universe.device)

        num_of_agents = indices.shape[0]

        # Randomize the push velocities every 10s
        if self._frame_idx % 4000 == 0:
            max_vel = self.randomizer_settings["push_range_xy"]
            velocities = torch.zeros((num_of_agents, 6), dtype=torch.float32)

            # Random values between [-max_vel, max_vel]
            velocities[:, :2] = (
                torch.rand((num_of_agents, 2), dtype=torch.float32) * 2 - 1
            ) * max_vel

            self._rigid_prim_view.set_velocities(velocities, indices)

    def on_reset(self, indices: Indices = None) -> None:
        super().on_reset(indices)

        if indices is None:
            indices = torch.arange(self._agent.num_agents)
        else:
            indices = indices.to(device=self._universe.device)

        num_to_reset = indices.shape[0]

        # set_world_poses is the only method that supports setting both positions and orientations
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

        joint_positions = torch.zeros((num_to_reset, 12), dtype=torch.float32)
        joint_velocities = torch.zeros_like(joint_positions)
        joint_efforts = torch.zeros_like(joint_positions)

        self._agent.joints_controller.reset(
            joint_positions,
            joint_velocities,
            joint_efforts,
            indices,
        )

        friction_range: list = self.randomizer_settings["friction_range"]
        dynamic_friction = (
            torch.rand((num_to_reset, 1), dtype=torch.float32)
            * (friction_range[1] - friction_range[0])
            + friction_range[0]
        )
        static_friction = (
            torch.rand((num_to_reset, 1), dtype=torch.float32)
            * (friction_range[1] - friction_range[0])
            + friction_range[0]
        )

        set_physics_properties(
            self.terrain.terrain_path,
            dynamic_friction=dynamic_friction,
            static_friction=static_friction,
        )

    def set_initial_positions(self, positions: torch.Tensor) -> None:
        self.initial_positions = positions.to(self._universe.device)

    def set_initial_orientations(self, orientations: torch.Tensor) -> None:
        self.initial_orientations = orientations.to(self._universe.device)
