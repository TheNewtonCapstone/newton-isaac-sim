from abc import ABC, abstractmethod
from typing import List, Optional

from core.agents import BaseAgent
from core.domain_randomizer import BaseDomainRandomizer
from core.terrain import BaseTerrainBuilder, BaseTerrainBuild
from core.types import EnvObservations, Actions, Indices
from core.universe import Universe


class BaseEnv(ABC):
    def __init__(
        self,
        agent: BaseAgent,
        num_envs: int,
        terrain_builders: List[BaseTerrainBuilder],
        domain_randomizer: BaseDomainRandomizer,
    ) -> None:
        self._universe: Optional[Universe] = None

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

        self._universe = universe

    @abstractmethod
    def step(
        self,
        actions: Actions,
    ) -> None:
        assert (
            self._is_constructed
        ), f"{self.__class__.__name__} not constructed: tried to step!"

        self._universe.step()

    @abstractmethod
    def reset(self, indices: Optional[Indices] = None) -> EnvObservations:
        assert (
            self._is_constructed
        ), f"{self.__class__.__name__} not constructed: tried to reset!"

        if indices is None:
            self._universe.reset()
            self._universe.step()

        return self.get_observations()

    @abstractmethod
    def get_observations(self) -> EnvObservations:
        assert (
            self._is_constructed
        ), f"{self.__class__.__name__} not constructed: tried to get observations!"

        return {}
