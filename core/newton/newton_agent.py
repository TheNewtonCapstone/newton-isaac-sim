from core.base.base_agent import BaseAgent


# this class describes how the agent will be constructed, nothing more
class NewtonAgent(BaseAgent):
    def __init__(self, config) -> None:
        super().__init__(config)

    def construct(self, root_path: str, world) -> str:
        super().construct(root_path, world)

        newton_prim_path = root_path + "/newton"

        # these only work after SimulationApp is initialized (to be done in scripts that import this class)
        import omni.isaac.core.utils.stage as stage_utils

        stage_utils.add_reference_to_stage(
            self.config["newton_usd_path"], prim_path=newton_prim_path
        )  # /<root_path>/newton

        world.reset()

        return newton_prim_path
