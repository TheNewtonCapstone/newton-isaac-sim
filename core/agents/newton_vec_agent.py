from core.agents import NewtonBaseAgent
from core.globals import TERRAINS_PATH, COLLISION_GROUPS_PATH
from core.logger import Logger
from core.types import EnvObservations, Actions


class NewtonVecAgent(NewtonBaseAgent):
    def construct(self) -> None:
        super().construct()

        self.base_path_expr = self.path + "/Newton_.*/base"
        root_path_expr = self.path + "/Newton_.*"
        self.path = self.path + "/Newton_0"

        import omni.isaac.core.utils.stage as stage_utils

        usd_path = "assets/newton/newton.usd"
        stage_utils.add_reference_to_stage(
            usd_path,
            prim_path=self.path,
        )
        Logger.info(
            f"Added reference agent from '{usd_path}' to '{self.path}' to the USD stage."
        )

        if self.num_agents > 1:
            Logger.info(f"Cloning {self.num_agents} agents from {self.path}.")

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

        Logger.info("NewtonVecAgent post-constructed.")

        self._is_post_constructed = True

    def step(self, actions: Actions) -> None:
        return super().step(actions)

    def get_observations(self) -> EnvObservations:
        return super().get_observations()
