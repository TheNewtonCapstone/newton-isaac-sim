from abc import ABC, abstractmethod
from typing import Optional

from core.globals import AGENTS_PATH
from core.types import Observations, Actions
from omni.isaac.core import World


class BaseAgent(ABC):
    def __init__(self, num_agents: int) -> None:
        self.path: str = ""
        self.num_agents: int = num_agents
        self.world: Optional[World] = None

        self._is_constructed: bool = False

    @abstractmethod
    def construct(self, world: World) -> None:
        self.path = AGENTS_PATH
        self.world = world

        from omni.isaac.core.utils.prims import create_prim

        create_prim(
            prim_path=self.path,
            prim_type="Scope",
        )

    @abstractmethod
    def step(self, actions: Actions) -> Observations:
        pass

    @abstractmethod
    def get_observations(self) -> Observations:
        pass
