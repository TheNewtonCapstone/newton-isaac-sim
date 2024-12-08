from abc import ABC, abstractmethod
from typing import List, Optional

from core.agents import BaseAgent
from core.domain_randomizer import BaseDomainRandomizer
from core.terrain import BaseTerrainBuilder, BaseTerrainBuild
from core.types import Observations, Actions, Indices
from core.universe import Universe


class BaseEnv(ABC):
    def __init__(
        self,
        agent: BaseAgent,
        num_envs: int,
        terrain_builders: List[BaseTerrainBuilder],
        domain_randomizer: BaseDomainRandomizer,
    ) -> None:
        self.universe: Optional[Universe] = None

        self.agent: BaseAgent = agent
        self.num_envs = num_envs

        self.terrain_builders: List[BaseTerrainBuilder] = terrain_builders
        self.terrain_builds: List[BaseTerrainBuild] = []

        self.domain_randomizer: BaseDomainRandomizer = domain_randomizer

        self._is_constructed = False

    @abstractmethod
    def construct(self, universe: Universe) -> None:
        assert (
            not self._is_constructed
        ), f"{self.__class__.__name__} already constructed!"

        self.universe = universe

    @abstractmethod
    def step(
        self,
        actions: Actions,
    ) -> Observations:
        assert (
            self._is_constructed
        ), f"{self.__class__.__name__} not constructed: tried to step!"

        self.universe.step()

        return self.get_observations()

    @abstractmethod
    def reset(self, indices: Indices = None) -> Observations:
        assert (
            self._is_constructed
        ), f"{self.__class__.__name__} not constructed: tried to reset!"

        if indices is None:
            self.universe.reset()
        else:
            self.universe.step()

        return self.get_observations()

    @abstractmethod
    def get_observations(self) -> Observations:
        assert (
            self._is_constructed
        ), f"{self.__class__.__name__} not constructed: tried to get observations!"

        return {}
