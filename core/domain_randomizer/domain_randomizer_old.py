import numpy as np

from core.base import BaseObject


class DomainRandomizer(BaseObject):
    def __init__(self, universe, num_envs, art_path, dr_configuration):
        super().__init__(universe)

        # Pass the universe instead of the universe
        # Pass the art_path from vec_agent
        # Creat the articulation view from the art_path

        import omni.replicator.isaac as dr
        import omni.replicator.core as rep

        self.dr = dr
        self.rep = rep

        self.universe = universe
        self.num_envs = num_envs
        self.art_path = art_path
        self.randomize = dr_configuration.get("randomize", False)
        self.dr_configuration = dr_configuration["randomization_params"]
        self.frequency = self.dr_configuration.get("frequency", 1)
        # self.domain_params = self.dr_configuration.get("twip", {})

        self.twip_art_view = self.dr.physics_view.create_articulation_view()

        self.num_dof = self.twip_art_view.num_dof
        self.rigid_body_names = self.twip_art_view.body_names
        print("rigid_body_names: ", self.rigid_body_names)

        self.on_interval_properties = {}
        self.on_reset_properties = {}

        self.format_dr_configuration()

        self.frame_idx = 0

        # Register the simulation context and articulation view
        self.dr.physics_view.register_simulation_context(self.universe)
        self.dr.physics_view.register_articulation_view(self.twip_art_view)
        print("Registered simulation context and articulation view")

    def construct(self):
        self._is_constructed = True

    def post_construct(self):
        self._is_post_constructed = True

    def format_dr_configuration(self):
        def process_property(distribution, range_values, body_type):
            range_str = self.get_randomization_range(range_values)

            if body_type == "dof_properties":
                if distribution == "uniform":
                    return self.rep.distribution.uniform(
                        tuple(range_str[0] * self.num_dof),
                        tuple(range_str[1] * self.num_dof),
                    )
                elif distribution == "normal":
                    return self.rep.distribution.normal(
                        tuple(range_str[0] * self.num_dof),
                        tuple(range_str[1] * self.num_dof),
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

        # Extract relevant sections from twip_params
        for gate_type in self.domain_params:
            gate_type_config = self.domain_params.get(gate_type, {})
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

        self.on_interval_properties = formatted_params.get("on_interval", {})
        self.on_reset_properties = formatted_params.get("on_reset", {})

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
                for body in self.on_interval_properties:
                    if "articulation_view_properties" in body:
                        for prop in self.on_interval_properties[body]:
                            body_properties = self.on_interval_properties.get(body, {})
                            args = body_properties.get(prop, {})

                            self.dr.physics_view.randomize_articulation_view(
                                view_name=self.twip_art_view.name,
                                operation=str(prop),
                                **args,
                            )
                    if "dof_properties" in body:
                        for prop in self.on_interval_properties[body]:
                            body_properties = self.on_interval_properties.get(body, {})
                            args = body_properties.get(prop, {})

                            self.dr.physics_view.randomize_articulation_view(
                                view_name=self.twip_art_view.name,
                                operation=str(prop),
                                **args,
                            )

            with self.dr.gate.on_env_reset():
                for body in self.on_reset_properties:
                    if "articulation_view_properties" in body:
                        for prop in self.on_reset_properties[body]:
                            body_properties = self.on_reset_properties.get(body, {})
                            args = body_properties.get(prop, {})

                            self.dr.physics_view.randomize_articulation_view(
                                view_name=self.twip_art_view.name,
                                operation=str(prop),
                                **args,
                            )
                    elif "dof_properties" in body:
                        for prop in self.on_reset_properties[body]:
                            body_properties = self.on_reset_properties.get(body, {})
                            args = body_properties.get(prop, {})

                            self.dr.physics_view.randomize_articulation_view(
                                view_name=self.twip_art_view.name,
                                operation=str(prop),
                                **args,
                            )

    def step_randomization(self):
        self.reset_inds = []
        if self.frame_idx % 200 == 0:
            self.reset_inds = np.arange(self.num_envs)
        self.dr.physics_view.step_randomization()
        self.frame_idx += 1
