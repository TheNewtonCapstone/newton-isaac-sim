from abc import ABC, abstractmethod
from typing import Optional

from core.globals import AGENTS_PATH
from core.types import Observations, Actions
from core.universe import Universe


class BaseAgent(ABC):
    def __init__(self, num_agents: int) -> None:
        assert num_agents > 0, f"Number of agents must be greater than 0: {num_agents}"

        self.path: str = ""
        self.num_agents: int = num_agents
        self._universe: Optional[Universe] = None

        self._is_constructed: bool = False

    @abstractmethod
    def construct(self, universe: Universe) -> None:
        assert (
            not self._is_constructed
        ), f"{self.__class__.__name__} already constructed: tried to construct!"

        self.path = AGENTS_PATH
        self._universe = universe

        from omni.isaac.core.utils.prims import create_prim

        create_prim(
            prim_path=self.path,
            prim_type="Scope",
        )

    @abstractmethod
    def step(self, actions: Actions) -> Observations:
        assert (
            self._is_constructed
        ), f"{self.__class__.__name__} not constructed: tried to step!"

        return self.get_observations()

    @abstractmethod
    def get_observations(self) -> Observations:
        assert (
            self._is_constructed
        ), f"{self.__class__.__name__} not constructed: tried to get observations!"

        return {}
