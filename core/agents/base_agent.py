from abc import ABC, abstractmethod
from typing import Dict

import torch
from core.types import Observations


class BaseAgent(ABC):
    def __init__(self, num_agents: int, world) -> None:
        from omni.isaac.core import World

        self.path: str = ""
        self.num_agents: int = num_agents
        self.world: World = world

        self._is_constructed: bool = False

    @abstractmethod
    def construct(self, root_path: str) -> str:
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Observations:
        pass

    @abstractmethod
    def reset(self) -> Observations:
        pass

    @abstractmethod
    def get_observations(self) -> Observations:
        pass
