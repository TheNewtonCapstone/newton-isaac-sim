from core.agents import NewtonBaseAgent
from core.globals import TERRAINS_PATH, COLLISION_GROUPS_PATH
from core.logger import Logger
from core.types import EnvObservations, Actions


class NewtonVecAgent(NewtonBaseAgent):
    def construct(self) -> None:
        super().construct()

        # path given by the parent class, where we spawn the agents
        agents_path = self.path

        # path where we'll load the first agent from file (needs to respect the below structure)
        load_path = f"{agents_path}/Newton_0"

        # real path is after the transform prim (i.e. ensures the agent is spawned at the correct height)
        self.path = f"{load_path}/_trans"

        # expression path to the base of all agents
        self.base_path_expr = f"{agents_path}/Newton_.*/_trans/base"

        # expression path to the root of all agents (i.e. right after the transform prim, basically the same as
        # self.path but as an expression)
        transformed_path_expr = f"{agents_path}/Newton_.*/_trans"

        import omni.isaac.core.utils.stage as stage_utils
        import omni.isaac.core.utils.prims as prim_utils

        usd_path = "assets/newton/newton.usd"
        stage_utils.add_reference_to_stage(
            usd_path,
            prim_path=load_path,
        )
        Logger.info(
            f"Added reference agent from '{usd_path}' to '{self.path}' to the USD stage."
        )

        # get the z position of the _trans prim of the agent
        prim = prim_utils.get_prim_at_path(self.path)
        self.transformed_position = prim.GetAttribute("xformOp:translate").Get()

        Logger.debug(
            f"NewtonVecAgent obtained transformed position: {self.transformed_position}"
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
        self.contact_sensor.register_self(
            f"{transformed_path_expr}/.*_LOWER_LEG_CONTACT"
        )

        self._is_constructed = True

    def post_construct(self) -> None:
        super().post_construct()

        Logger.info("NewtonVecAgent post-constructed.")

        self._is_post_constructed = True

    def step(self, actions: Actions) -> None:
        return super().step(actions)

    def get_observations(self) -> EnvObservations:
        return super().get_observations()
