from core.agents import NewtonBaseAgent
from core.globals import TERRAINS_PATH, COLLISION_GROUPS_PATH
from core.types import Observations, Actions
from core.universe import Universe


class NewtonVecAgent(NewtonBaseAgent):
    def construct(self, universe: Universe) -> None:
        super().construct(universe)

        self.base_path_expr = self.path + "/Newton_*/base"
        self.path = self.path + "/Newton_0"

        import omni.isaac.core.utils.stage as stage_utils

        stage_utils.add_reference_to_stage(
            "assets/newton/newton.usd",
            prim_path=self.path,
        )

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

        self.imu.construct(self.base_path_expr)
        self.joints_controller.construct(self.base_path_expr)

        self._universe.reset()

        self._is_constructed = True

    def step(self, actions: Actions) -> None:
        return super().step(actions)

    def get_observations(self) -> Observations:
        return super().get_observations()
