from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from core.agents import BaseAgent
from core.domain_randomizer.domain_randomizer import DomainRandomizer
from core.terrain import TerrainBuilder, TerrainBuild
from core.types import Settings, Observations


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

        self.path: str = ""
        self.num_envs = num_envs
        self.agent: BaseAgent = agent
        self.world: Optional[World] = None
        self.world_settings = world_settings

        self.terrain_builders: List[TerrainBuilder] = terrain_builders
        self.terrain_paths: List[TerrainBuild] = []

        self.domain_randomizer: DomainRandomizer = None  # TODO
        self.randomizer_settings: Settings = randomizer_settings

        self._time = 0

    def __del__(self):
        if self.world is not None:
            self.world.stop()

    @abstractmethod
    def construct(self) -> None:
        from omni.isaac.core import World

        self.world: World = World(
            physics_dt=self.world_settings["physics_dt"],
            rendering_dt=self.world_settings["rendering_dt"],
            stage_units_in_meters=self.world_settings["stage_units_in_meters"],
            backend=self.world_settings["backend"],
            device=self.world_settings["device"],
        )

        from core.utils.gpu import get_free_gpu_memory

        free_device_memory = get_free_gpu_memory()
        assert free_device_memory > 0, "No free GPU memory found"

        # Adjust physics scene settings (mainly for GPU memory allocation)
        phys_context = self.world.get_physics_context()
        phys_context.set_gpu_found_lost_aggregate_pairs_capacity(
            free_device_memory // 5 * 3
        )  # there should be more contacts than overall pairs
        phys_context.set_gpu_total_aggregate_pairs_capacity(free_device_memory // 5 * 2)

        self.path = f"/{self.__class__.__name__.capitalize()}"

        return self.path

    @abstractmethod
    def step(
        self,
        actions: torch.Tensor,
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
    def reset(
        self,
    ) -> Observations:
        return self.get_observations()

    @abstractmethod
    def get_observations(self) -> Observations:
        return {}
