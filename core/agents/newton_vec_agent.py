from core.agents import NewtonBaseAgent
from core.globals import TERRAINS_PATH, PHYSICS_SCENE_PATH, COLLISION_GROUPS_PATH
from core.types import Observations, Actions


class NewtonVecAgent(NewtonBaseAgent):
    def construct(self, root_path: str) -> str:
        if self._is_constructed:
            return self.path

        super().construct(root_path)

        self.path = root_path + "/newton_0"

        import omni.isaac.core.utils.stage as stage_utils

        stage_utils.add_reference_to_stage(
            "assets/newton/newton.usd",
            prim_path=self.path,
        )

        self.world.reset()  # ensures that the USD is loaded

        from omni.isaac.cloner import Cloner

        cloner = Cloner()
        cloner.define_base_env(self.path)

        agent_paths = cloner.generate_paths(self.path[:-1], self.num_agents)

        cloner.filter_collisions(
            physicsscene_path=PHYSICS_SCENE_PATH,
            collision_root_path=COLLISION_GROUPS_PATH,
            prim_paths=agent_paths,
            global_paths=[TERRAINS_PATH],
        )
        cloner.clone(
            source_prim_path=self.path,
            prim_paths=agent_paths,
        )

        from omni.isaac.core.articulations import ArticulationView

        newton_art_view = ArticulationView(
            prim_paths_expr=self.path + "*/newton/base",
            name="newton_art_view",
        )
        self.world.scene.add(newton_art_view)

        self._construct_imu(self.path + "*/newton/base")
        self._construct_joints_controller(self.path + "*/newton/base")

        self.world.reset()

        self._is_constructed = True

        return self.path

    def step(self, actions: Actions) -> Observations:
        super().step(actions)

        # TODO: self.joints_controller.update(actions)

        self.joints_controller.step()

        return self.get_observations()

    def reset(self) -> Observations:
        super().reset()

        return self.get_observations()

    def get_observations(self) -> Observations:
        return super().get_observations()
