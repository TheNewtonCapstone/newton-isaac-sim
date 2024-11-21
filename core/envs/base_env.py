import weakref
from abc import ABC, abstractmethod
from typing import List, Optional

import omni.timeline
import torch
from core.agents import BaseAgent
from core.domain_randomizer.domain_randomizer import DomainRandomizer
from core.globals import PHYSICS_SCENE_PATH, TERRAINS_PATH, LIGHTS_PATH
from core.terrain import TerrainBuilder, TerrainBuild
from core.types import Settings, Observations, Actions


class BaseEnv(ABC):
    def __init__(
        self,
        agent: BaseAgent,
        num_envs: int,
        terrain_builders: List[TerrainBuilder],
        world_settings: Settings,
        randomizer_settings: Settings,
    ) -> None:
        from omni.isaac.core import World

        self.num_envs = num_envs
        self.agent: BaseAgent = agent
        self.world: Optional[World] = None
        self.world_settings = world_settings

        self.terrain_builders: List[TerrainBuilder] = terrain_builders
        self.terrain_builds: List[TerrainBuild] = []

        self.domain_randomizer: DomainRandomizer = None  # TODO
        self.randomizer_settings: Settings = randomizer_settings

        self._time = 0

    def __del__(self):
        if self.world is not None:
            self.world.stop()

    @abstractmethod
    def construct(self) -> None:
        from core.utils.gpu import get_free_gpu_memory

        free_device_memory = get_free_gpu_memory()
        assert free_device_memory > 0, "No free GPU memory found"

        from omni.isaac.core import World

        self.world: World = World(
            physics_prim_path=PHYSICS_SCENE_PATH,
            physics_dt=self.world_settings["physics_dt"],
            rendering_dt=self.world_settings["rendering_dt"],
            stage_units_in_meters=self.world_settings["stage_units_in_meters"],
            backend=self.world_settings["backend"],
            device=self.world_settings["device"],
            sim_params=self.world_settings["sim_params"],
        )

        self.world.reset()
        self.world.get_physics_context().enable_ccd(True)

        # ensures some base prims exist

        from omni.isaac.core.utils.prims import create_prim

        create_prim(
            prim_path=LIGHTS_PATH,
            prim_type="Scope",
        )

        create_prim(
            prim_path=TERRAINS_PATH,
            prim_type="Scope",
        )

    @abstractmethod
    def step(
        self,
        actions: Actions,
        render: bool,
    ) -> Observations:

        # From IsaacLab (SimulationContext)
        # need to do one update to refresh the app
        # reason: physics has to parse the scene again and inform other extensions like hydra-delegate.
        # without this the app becomes unresponsive. If render is True, the world updates the app automatically.
        if not render:
            self.world.app.update()

        self.world.step(render=render)
        return self.get_observations()

    @abstractmethod
    def reset(self) -> Observations:
        return self.get_observations()

    @abstractmethod
    def get_observations(self) -> Observations:
        return {}
