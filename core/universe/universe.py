from core.globals import PHYSICS_SCENE_PATH, LIGHTS_PATH, PHYSICS_PATH
from core.types import Settings
from omni.isaac.kit import SimulationApp
from omni.isaac.core import SimulationContext


class Universe(SimulationContext):
    def __init__(
        self,
        headless: bool,
        sim_app: SimulationApp,
        world_settings: Settings,
    ):
        self.sim_app: SimulationApp = sim_app
        self._world_settings: Settings = world_settings

        self._headless = headless

        self._is_constructed: bool = False

        super().__init__(
            physics_prim_path=PHYSICS_SCENE_PATH,
            physics_dt=self._world_settings["physics_dt"],
            rendering_dt=self._world_settings["rendering_dt"],
            stage_units_in_meters=self._world_settings["stage_units_in_meters"],
            backend=self._world_settings["backend"],
            device=self._world_settings["device"],
            sim_params=self._world_settings["sim_params"],
        )

    def __del__(self):
        if self._is_constructed:
            self._world.stop()

    @property
    def headless(self) -> bool:
        return self._headless

    @property
    def physics_world(self):
        return self._world

    @property
    def is_playing(self) -> bool:
        return self.sim_app.is_running()

    def add_to_scene(self, prim):
        """

        Args:
            prim (XFormPrim): The prim to add to the scene: can be any as supported by the Scene.add(obj) method
        """
        # self.scene.add(prim)

    def construct(self) -> None:
        assert (
            not self._is_constructed
        ), "Universe already constructed: tried to construct!"

        # this needs to be created before any other omniverse-related imports
        self.sim_app = SimulationApp(
            {
                "headless": self.headless,
            },
            experience="./apps/omni.isaac.sim.newton.kit",
        )

        from core.utils.gpu import get_free_gpu_memory

        free_device_memory = get_free_gpu_memory()
        assert free_device_memory > 1000, "No free GPU memory found (>1000)"

        from omni.isaac.core.utils.stage import clear_stage, add_reference_to_stage

        # ensures we have no conflicting stages
        clear_stage()

        # adds a saved PhysicsScene prim
        add_reference_to_stage(
            "assets/physics.usd",
            prim_path=PHYSICS_PATH,
        )

        # adds a default lighting setup
        add_reference_to_stage(
            "assets/lights.usd",
            prim_path=LIGHTS_PATH,
        )

        from omni.isaac.core import World

        # TODO: if we're training, disable the scene_query_support option in sim_params

        phys_ctx = self.get_physics_context()
        # the following are performance options, that we don't care about when not in headless mode
        # phys_ctx.enable_fabric(self.headless)
        phys_ctx.set_physx_update_transformations_settings(
            update_to_usd=not self.headless,
            update_velocities_to_usd=not self.headless,
        )

        self._is_constructed = True

    def step(self) -> None:
        assert self._is_constructed, "Universe not constructed: tried to step!"

        if not self.sim_app.is_running():
            exit(
                0
            )  # TODO: make this optional and more graceful (probably in a task callback)

        # From IsaacLab (SimulationContext)
        # need to do one update to refresh the app
        # reason: physics has to parse the scene again and inform other extensions like hydra-delegate.
        # without this the app becomes unresponsive. If render is True, the world updates the app automatically.
        if self.headless:
            self._world.app.update()

        self._world.step(render=not self.headless)

    def reset(self):
        assert self._is_constructed, "Universe not constructed: tried to reset!"

        import omni.log

        # TODO: implement proper logging, probably with a logger class and different channels for different things (
        #  e.g. rl, physics, etc.)
        omni.log.info("Universe reset", "newton.core.universe")

        self._world.reset()
