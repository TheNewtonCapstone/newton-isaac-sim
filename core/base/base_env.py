from abc import ABC, abstractmethod
from typing import Dict

import GPUtil
import torch
from core.base.base_agent import BaseAgent
from core.domain_randomizer.domain_randomizer import DomainRandomizer
from core.terrain.terrain import TerrainBuilder


# TODO: separate into 3 classes: BaseEnv, BaseTask, BaseAgent
# BaseEnv: contains the world, agents and settings
# BaseTask: contains the world (with its agents) task settings and the task logic
# BaseAgent: contains the agent settings and the agent logic
# The goal is to make it so that any task can be run in any environment with any agent
# Need to figure out how abc works in python and how to use it to enforce the structure of the classes


class BaseEnv(ABC):
    def __init__(
        self,
        world_settings: Dict,
        num_envs: int,
        terrain_builders: list[TerrainBuilder],
        randomization_settings: Dict,
    ) -> None:
        self.world = None
        self.agent = None
        self.terrain_builders = terrain_builders
        self.terrain_paths = []
        self.randomization_settings = randomization_settings
        self.world_settings = world_settings
        self.num_envs = num_envs

        self.domain_randomizer: DomainRandomizer = None
        self.randomize = randomization_settings.get("randomize", False)
        self.randomization_params = randomization_settings.get(
            "randomization_params", {}
        )

    @abstractmethod
    def construct(self, agent: BaseAgent) -> str:
        self.agent = agent  # save the agent class for informative purposes (i.e. introspection/debugging)

        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import Sdf, UsdLux

        stage = get_current_stage()

        self.world: World = World(
            physics_dt=self.world_settings["physics_dt"],
            rendering_dt=self.world_settings["rendering_dt"],
            stage_units_in_meters=self.world_settings["stage_units_in_meters"],
            backend=self.world_settings["backend"],
            device=self.world_settings["device"],
        )

        devices = GPUtil.getGPUs()
        assert len(devices) > 0, "No GPU devices found"

        main_device: GPUtil.GPU = devices[0]
        free_device_memory = main_device.memoryFree
        assert free_device_memory > 0, "No free GPU memory found"

        # Adjust physics scene settings (mainly for GPU memory allocation)
        phys_context = self.world.get_physics_context()
        phys_context.set_gpu_found_lost_aggregate_pairs_capacity(
            free_device_memory // 5 * 3
        )  # there should be more contacts than overall pairs
        phys_context.set_gpu_total_aggregate_pairs_capacity(free_device_memory // 5 * 2)

        # Add sun
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/distantLight"))
        sun.CreateIntensityAttr(500)

        return None

    @abstractmethod
    def step(self, actions: torch.Tensor, render: bool) -> torch.Tensor:
        self.world.step(render=render)
        return torch.zeros(0)

    @abstractmethod
    def reset(
        self,
        indices: torch.LongTensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        self.world.reset()
        return {"obs": torch.zeros(0)}
