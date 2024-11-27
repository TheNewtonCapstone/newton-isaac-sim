from abc import abstractmethod

from core.agents import NewtonBaseAgent
from core.globals import TERRAINS_PATH, PHYSICS_SCENE_PATH, COLLISION_GROUPS_PATH
from core.types import Observations, Actions
from omni.isaac.core import World


class NewtonVecAgent(NewtonBaseAgent):
    def construct(self, world: World) -> None:
        super().construct(world)

        self.base_path_expr = self.path + "/Newton_*/base"
        self.path = self.path + "/Newton_0"

        import omni.isaac.core.utils.stage as stage_utils

        stage_utils.add_reference_to_stage(
            "assets/newton/newton.usd",
            prim_path=self.path,
        )

        self.world.reset()  # ensures that the USD is loaded

        from omni.isaac.cloner import Cloner

        cloner = Cloner()
        cloner.define_base_env(self.path)

        agent_paths = cloner.generate_paths(self.path[:-2], self.num_agents)

        cloner.filter_collisions(
            prim_paths=agent_paths,
            physicsscene_path=PHYSICS_SCENE_PATH,
            collision_root_path=COLLISION_GROUPS_PATH,
            global_paths=[TERRAINS_PATH],
        )
        cloner.clone(
            source_prim_path=self.path,
            prim_paths=agent_paths,
            replicate_physics=True,
        )

        self.world.reset()

        self._construct_imu(self.base_path_expr)
        self._construct_joints_controller(self.base_path_expr)

        self.world.reset()

        self._is_constructed = True

    def step(self, actions: Actions) -> None:
        return super().step(actions)

    def get_observations(self) -> Observations:
        return super().get_observations()
