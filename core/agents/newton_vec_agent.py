from core.agents import NewtonBaseAgent
from core.globals import TERRAINS_PATH, COLLISION_GROUPS_PATH
from core.types import EnvObservations, Actions


class NewtonVecAgent(NewtonBaseAgent):
    def construct(self) -> None:
        super().construct()

        self.base_path_expr = self.path + "/Newton_.*/base"
        root_path_expr = self.path + "/Newton_.*"
        self.path = self.path + "/Newton_0"

        import omni.isaac.core.utils.stage as stage_utils

        stage_utils.add_reference_to_stage(
            "assets/newton/newton.usd",
            prim_path=self.path,
        )

        if self.num_agents > 1:
            from omni.isaac.cloner import Cloner

            cloner = Cloner()
            cloner.define_base_env(self.path)

            agent_paths = cloner.generate_paths(self.path[:-2], self.num_agents)

            cloner.filter_collisions(
                prim_paths=agent_paths,
                physicsscene_path=self._universe.get_physics_context().prim_path,
                collision_root_path=COLLISION_GROUPS_PATH,
                global_paths=[TERRAINS_PATH],
            )
            cloner.clone(
                source_prim_path=self.path,
                prim_paths=agent_paths,
                copy_from_source=True,
            )

        self.imu.register_self(self.base_path_expr)
        self.joints_controller.register_self(self.base_path_expr)
        self.contact_sensor.register_self(root_path_expr + "/.*_LOWER_LEG_CONTACT")

        self._is_constructed = True

    def post_construct(self) -> None:
        super().post_construct()

        self._is_post_constructed = True

    def step(self, actions: Actions) -> None:
        return super().step(actions)

    def get_observations(self) -> EnvObservations:
        return super().get_observations()
