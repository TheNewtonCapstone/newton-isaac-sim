import torch

from torch import Tensor

from core.agents import NewtonBaseAgent
from core.globals import TERRAINS_PATH, PHYSICS_SCENE_PATH, COLLISION_GROUPS_PATH
from core.types import Observations, Actions
from omni.isaac.core import World


class NewtonVecAgent(NewtonBaseAgent):
    def construct(self, world: World) -> None:
        if self._is_constructed:
            return

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

        self.newton_art_view = ArticulationView(
            prim_paths_expr=self.base_path_expr,
            name="newton_art_view",
        )
        self.world.scene.add(self.newton_art_view)

        self._construct_imu(self.base_path_expr)
        self._construct_joints_controller(self.base_path_expr)

        self.world.reset()

        self._is_constructed = True

    def step(self, actions: Actions) -> Observations:
        super().step(actions)

        # TODO: self.joints_controller.update(actions)

        self.joints_controller.step()
        self.imu.step()

        return self._get_observations()

    def _get_observations(self) -> Observations:
        # TODO: self.imu

        return super()._get_observations()
