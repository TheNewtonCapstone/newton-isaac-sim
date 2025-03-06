from typing import Optional

import genesis as gs

from ..logger import Logger
from ..types import Config, Mode


# TODO: Centralize configuration data into the Universe class
#   We should have a single source of truth for all configuration, including the simulation app settings, world settings,
#   and any other settings that are used in the simulation. This will make it easier to manage and query the settings.


class Universe:
    def __init__(
        self,
        universe_config: Config,
        headless: bool,
        num_envs: int,
        mode: Mode,
        run_name: Optional[str],
        ros_enabled: bool = False,
    ):
        self._universe_config: Config = universe_config

        self._num_envs: int = num_envs
        self._headless: bool = headless
        self._mode: Mode = mode
        self._run_name: Optional[str] = run_name
        self._ros_enabled: bool = ros_enabled

        # Device
        selected_backend = gs.gs_backend.cpu
        if self.device.startswith("cuda") or self.device.startswith("gpu"):
            selected_backend = gs.gs_backend.gpu

        gs.init(backend=selected_backend)

        # Scene
        self._scene: gs.Scene

        self._create_scene()

    @property
    def device(self) -> str:
        return self._universe_config["sim_options"]["device"]

    @property
    def headless(self) -> bool:
        return self._headless

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def run_name(self) -> Optional[str]:
        return self._run_name

    @property
    def ros_enabled(self) -> bool:
        return self._ros_enabled

    @property
    def physics_dt(self) -> float:
        return self._universe_config["sim_options"]["physics_dt"]

    @property
    def control_dt(self) -> float:
        return self._universe_config["sim_options"]["control_dt"]

    @property
    def gravity(self) -> float:
        return self._universe_config["sim_options"]["gravity"]

    @property
    def current_time(self) -> float:
        return self.current_timestep * self.physics_dt

    @property
    def current_timestep(self) -> int:
        return self._scene.t

    @property
    def scene(self) -> gs.Scene:
        return self._scene

    def step(self, render: bool = False) -> None:
        self._scene.step(render and not self.headless)

    def reset(self) -> None:
        Logger.info("Resetting universe")

        self._scene.reset()

    def build(self) -> None:
        Logger.info(f"Building universe with {self.num_envs} environments")

        self._scene.build(
            num_envs=self._num_envs,
            center_envs_at_origin=True,
            env_spacing=(1.0, 1.0),
        )

    def _create_scene(self) -> None:
        sim_options = gs.options.SimOptions(
            dt=self.physics_dt,
            substeps=self._universe_config["sim_options"]["substeps"],
            gravity=(0, 0, self.gravity),
        )
        viewer_options = gs.options.ViewerOptions()
        vis_options = gs.options.VisOptions()

        self._scene = gs.Scene(
            show_FPS=False,
            show_viewer=not self.headless,
            sim_options=sim_options,
            viewer_options=viewer_options,
            vis_options=vis_options,
        )
