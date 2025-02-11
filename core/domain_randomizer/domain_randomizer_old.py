from typing import Optional
import torch
import numpy as np
from core.logger import Logger

from omni.isaac.core.articulations import ArticulationView
from core.base import BaseObject
from ..types import Config, Indices


class DomainRandomizer(BaseObject):
    def __init__(self, universe, agent, num_envs, dr_configuration):
        super().__init__(universe)

        # Pass the universe instead of the universe
        # Pass the art_path from vec_agent
        # Creat the articulation view from the art_path

        import omni.replicator.isaac as dr
        import omni.replicator.core as rep

        self.dr = dr
        self.rep = rep

        self._universe = universe
        self._agent = agent
        self.num_envs = num_envs
        self.randomize = dr_configuration.get("randomize", False)
        self.dr_configuration = dr_configuration["randomization_params"]
        self.frequency = dr_configuration.get("frequency")
        self._art_path = self._agent.base_path_expr

        # Construct the domain randomizer
        self._newton_art_view = None
        self._num_dof = 0
        self._rigid_body_names = []

        # Domain randomization properties
        self._on_interval_properties = {}
        self._on_reset_properties = {}

        from core.utils.math import IDENTITY_QUAT

        self.reset_newton_positions: torch.Tensor = torch.zeros((self.num_envs, 3))
        self.reset_newton_orientations: torch.Tensor = IDENTITY_QUAT.repeat(
            self.num_envs, 1
        )
        self.initial_positions: torch.Tensor = torch.zeros(
            (self.num_envs, 3), device=self._universe.device
        )
        self.initial_orientations: torch.Tensor = torch.zeros(
            (self.num_envs, 4), device=self._universe.device
        )

        self._frame_idx = 0

    def construct(self):

        # Create the articulation view
        self._newton_art_view = ArticulationView(self._art_path)
        self._universe.add_prim(self._newton_art_view)

        self._is_constructed = True

    def post_construct(self):

        self._num_dof = self._newton_art_view.num_dof
        self._rigid_body_names = self._newton_art_view.body_names

        # Register the simulation context and articulation view
        self.dr.physics_view.register_simulation_context(self._universe)
        self.dr.physics_view.register_articulation_view(self._newton_art_view)
        Logger.info(f"DomainRandomizer constructed with {self.num_envs} environments")

        # Format the domain randomization configuration
        self.format_dr_configuration()
        if self.randomize:
            self.apply_randomization()

        self._is_post_constructed = True

    def on_step(self):
        self.step_randomization()

    def on_reset(self, indices: Indices = None):
        if indices is None:
            indices = torch.arange(self.num_envs)
        else:
            indices = indices.to(device=self._universe.device)

        num_to_reset = indices.shape[0]

        # set_world_poses is the only method that supports setting both positions and orientations
        self._newton_art_view.set_world_poses(
            positions=self.initial_positions[indices],
            orientations=self.initial_orientations[indices],
            indices=indices,
            usd=self._universe.use_usd_physics,
        )

        # using set_velocities instead of individual methods (lin & ang),
        # because it's the only method supported in the GPU pipeline (default pipeline)
        self._newton_art_view.set_velocities(
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

    def format_dr_configuration(self):
        def process_property(distribution, range_values, body_type):
            range_str = self.get_randomization_range(range_values)

            if body_type == "dof_properties":
                if distribution == "uniform":
                    return self.rep.distribution.uniform(
                        tuple(range_str[0] * self._num_dof),
                        tuple(range_str[1] * self._num_dof),
                    )
                elif distribution == "normal":
                    return self.rep.distribution.normal(
                        tuple(range_str[0] * self._num_dof),
                        tuple(range_str[1] * self._num_dof),
                    )
                else:
                    raise ValueError(f"Invalid distribution type: {distribution}")

            else:
                if distribution == "uniform":
                    return self.rep.distribution.uniform(
                        tuple(range_str[0]), tuple(range_str[1])
                    )
                elif distribution == "normal":
                    return self.rep.distribution.normal(
                        tuple(range_str[0]), tuple(range_str[1])
                    )
                else:
                    raise ValueError(f"Invalid distribution type: {distribution}")

        def format_properties(properties, body_type):
            return {
                prop: process_property(
                    prop_data.get("distribution", "uniform"),
                    prop_data.get("range", []),
                    body_type,
                )
                for prop, prop_data in properties.items()
            }

        formatted_params = {}

        for gate_type in self.dr_configuration:
            gate_type_config = self.dr_configuration.get(gate_type, {})
            formatted_params[gate_type] = {}

            for property_type in [
                "articulation_view_properties",
                "dof_properties",
                "rigid_body_properties",
            ]:
                property_config = gate_type_config.get(property_type, {})

                formatted_params[gate_type][property_type] = {
                    "additive": format_properties(
                        property_config.get("additive", {}), property_type
                    ),
                    "scaling": format_properties(
                        property_config.get("scaling", {}), property_type
                    ),
                    "direct": format_properties(
                        property_config.get("direct", {}), property_type
                    ),
                }

        self._on_interval_properties = formatted_params.get("on_interval", {})
        self._on_reset_properties = formatted_params.get("on_reset", {})

    def get_randomization_range(self, prop_range):
        from_x = []
        to_y = []
        if isinstance(prop_range[0], list):
            for item in prop_range:
                from_x.append(item[0])
                to_y.append(item[1])
        else:
            from_x = [prop_range[0]]
            to_y = [prop_range[1]]

        return from_x, to_y

    def apply_randomization(self):
        with self.dr.trigger.on_rl_frame(num_envs=self.num_envs):
            with self.dr.gate.on_interval(interval=self.frequency):
                for body in self._on_interval_properties:
                    if "articulation_view_properties" in body:
                        for prop in self._on_interval_properties[body]:
                            body_properties = self._on_interval_properties.get(body, {})
                            args = body_properties.get(prop, {})

                            self.dr.physics_view.randomize_articulation_view(
                                view_name=self._newton_art_view.name,
                                operation=str(prop),
                                **args,
                            )
                    if "dof_properties" in body:
                        for prop in self._on_interval_properties[body]:
                            body_properties = self._on_interval_properties.get(body, {})
                            args = body_properties.get(prop, {})

                            self.dr.physics_view.randomize_articulation_view(
                                view_name=self._newton_art_view.name,
                                operation=str(prop),
                                **args,
                            )

    def step_randomization(self):
        reset_inds = []
        if self._frame_idx % 200 == 0:
            reset_inds = np.arange(self.num_envs)
        self.dr.physics_view.step_randomization()
        self._frame_idx += 1
