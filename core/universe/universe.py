from typing import Any

from core.globals import LIGHTS_PATH, PHYSICS_PATH
from core.types import Settings
from omni.isaac.core import SimulationContext
from omni.isaac.core.prims import XFormPrim, XFormPrimView
from omni.isaac.core.scenes import Scene
from omni.isaac.kit import SimulationApp
from pxr import Usd


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

        if self.has_gui:
            from omni.kit.viewport.utility import get_active_viewport
            from omni.kit.widget.viewport.api import ViewportAPI

            viewport_ctx: ViewportAPI = get_active_viewport()
            viewport_ctx.updates_enabled = True

        # Scene
        self._scene = Scene()

        super().__init__(
            physics_prim_path=PHYSICS_PATH,
            physics_dt=self._world_settings["physics_dt"],
            rendering_dt=self._world_settings["rendering_dt"],
            stage_units_in_meters=self._world_settings["stage_units_in_meters"],
            backend=self._world_settings["backend"],
            device=self._world_settings["device"],
            sim_params=self._world_settings["sim_params"],
        )

    @property
    def headless(self) -> bool:
        return self._headless

    @property
    def has_gui(self) -> bool:
        return not self.headless

    @property
    def use_fabric_physics(self) -> bool:
        return self._world_settings["sim_params"]["use_fabric"]

    @property
    def use_usd_physics(self) -> bool:
        return not self.use_fabric_physics

    def step(self, render: bool = False) -> None:
        """Steps the simulation. This behaves exactly like IsaacLab SimulationContext's step function.

        .. note::
            This function blocks if the timeline is paused. It only returns when the timeline is playing.

        Args:
            render: Unused, rendering depends on the headless flag.
        """

        # check if the simulation timeline is paused. in that case keep stepping until it is playing
        if not self.is_playing():
            # step the simulator (but not the physics) to have UI still active
            while not self.is_playing():
                self.render()

                # meantime if someone stops, break out of the loop
                if self.is_stopped():
                    break

            # need to do one step to refresh the app
            # reason: physics has to parse the scene again and inform other extensions like hydra-delegate.
            #   without this the app becomes unresponsive.
            self.update_app_no_sim()

        # step the simulation
        super().step(render=self.has_gui)

    def render(self) -> None:
        if self.has_gui:
            # if we have a viewport, render the scene as it should by default
            super().render()
            return

        # otherwise, we need to manually update the app (without stepping the simulation)
        self.update_app_no_sim()

    def reset(self, soft: bool = False) -> None:
        import omni.log

        # TODO: implement proper logging, probably with a logger class and different channels for different things (
        #  e.g. rl, physics, etc.)
        omni.log.info("Universe reset", "newton.core.universe")

        # From Isaac Lab (SimulationContext): https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/sim/simulation_context.py#L423
        # perform additional rendering steps to warm up replicator buffers
        # this is only needed for the first time we set the simulation
        if not soft:
            if not self.is_stopped():
                self.stop()

            self.initialize_physics()

            # ensures all the "_scene.add" calls are processed
            self._scene._finalize(self.physics_sim_view)
            self._scene.post_reset()

            for _ in range(2):
                self.render()

    def add_prim(self, prim: XFormPrim | XFormPrimView):
        self._scene.add(prim)

    def update_app_no_sim(self) -> None:
        self.set_carb_setting("/app/player/playSimulations", False)
        self.app.update()
        self.set_carb_setting("/app/player/playSimulations", True)

    def set_carb_setting(self, name: str, value: Any):
        """Set simulation settings using the Carbonite SDK.

        .. note::
            If the input setting name does not exist, it will be created. If it does exist, the value will be
            overwritten. Please make sure to use the correct setting name.

            To understand the settings interface, please refer to the
            `Carbonite SDK <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/settings.html>`_
            documentation.

        Args:
            name: The name of the setting.
            value: The value of the setting.
        """
        self._settings.set(name, value)

    def get_carb_setting(self, name: str) -> Any:
        """Read the simulation setting using the Carbonite SDK.

        Args:
            name: The name of the setting.

        Returns:
            The value of the setting.
        """
        return self._settings.get(name)

    # Helpers

    def _init_stage(self, *args, **kwargs) -> Usd.Stage:
        from core.utils.gpu import get_free_gpu_memory

        free_device_memory = get_free_gpu_memory()
        assert free_device_memory >= 1000, "Not enough GPU memory free (<1000)"

        _ = super()._init_stage(
            *args,
            **kwargs,
        )

        # Physics-related methods may now be used

        from omni.isaac.core.utils.stage import add_reference_to_stage

        # adds a default lighting setup
        add_reference_to_stage(
            "assets/lights.usd",
            prim_path=LIGHTS_PATH,
        )

        # a stage update here is needed for the case when physics_dt != rendering_dt, otherwise the app crashes
        # when in headless mode
        self.update_app_no_sim()

        # set additional physx parameters and bind material
        self._set_additional_physx_params()

        # return the stage
        return self.stage

    async def _initialize_stage_async(self, *args, **kwargs) -> Usd.Stage:
        # avoids usage within the Omniverse Extensions workflow
        assert (
            False
        ), "Async initialization is not supported: please run this in Standalone mode!"

    def _set_additional_physx_params(self):
        from pxr import PhysxSchema

        # obtain the physics scene api
        physics_scene_prim = self.get_physics_context().get_current_physics_scene_prim()
        physx_scene_api: PhysxSchema.PhysxSceneAPI = PhysxSchema.PhysxSceneAPI(
            physics_scene_prim
        )

        # assert that scene api is not None
        assert physx_scene_api is not None, "Physics scene API is None!"

        # set parameters not directly supported by the SimulationContext constructor
        # -- Continuous Collision Detection (CCD)
        # ref: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html?highlight=ccd#continuous-collision-detection
        self.get_physics_context().enable_ccd(
            self._world_settings["sim_params"]["enable_ccd"]
        )

        # -- Gravity
        self.get_physics_context().set_gravity(-9.81)

        # -- Global Contact Processing
        self.set_carb_setting(
            "/physics/disableContactProcessing",
            self._world_settings["sim_params"]["disable_contact_processing"],
        )

        # -- Improved determinism by PhysX
        physx_scene_api.CreateEnableEnhancedDeterminismAttr(
            self._world_settings["sim_params"]["enable_enhanced_determinism"]
        )

        # -- Position iteration count
        physx_scene_api.CreateMinPositionIterationCountAttr(
            self._world_settings["sim_params"]["min_position_iteration_count"]
        )
        physx_scene_api.CreateMaxPositionIterationCountAttr(
            self._world_settings["sim_params"]["max_position_iteration_count"]
        )

        # -- Velocity iteration count
        physx_scene_api.CreateMinVelocityIterationCountAttr(
            self._world_settings["sim_params"]["min_velocity_iteration_count"]
        )
        physx_scene_api.CreateMaxVelocityIterationCountAttr(
            self._world_settings["sim_params"]["max_velocity_iteration_count"]
        )

        # create the default physics material
        # this material is used when no material is specified for a primitive
        # check: https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/simulation-control/physics-settings.html#physics-materials

        from core.utils.physics import set_physics_properties

        set_physics_properties(
            self.get_physics_context().prim_path,
            static_friction=self._world_settings["defaults"]["physics_material"][
                "static_friction"
            ],
            dynamic_friction=self._world_settings["defaults"]["physics_material"][
                "dynamic_friction"
            ],
            restitution=self._world_settings["defaults"]["physics_material"][
                "restitution"
            ],
        )

        self.update_app_no_sim()
