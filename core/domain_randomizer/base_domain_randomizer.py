from abc import ABC, abstractmethod
from typing import List

from core.agents import BaseAgent
from core.terrain import BaseTerrainBuild
from core.types import Settings, Indices


class BaseDomainRandomizer(ABC):
    def __init__(
        self,
        seed: int,
        agent: BaseAgent,
        # TODO: think about how to construct the domain randomizers appropriately (without circular imports)
        terrain_builds: List[BaseTerrainBuild],
        randomizer_settings: Settings,
    ):
        self.seed: int = seed

        self._agent: BaseAgent = agent
        self._terrain_builds: List[BaseTerrainBuild] = terrain_builds

        self.randomizer_settings: Settings = randomizer_settings

        self._time: int = 0
        self._is_constructed: bool = False

    @abstractmethod
    def construct(self) -> None:
        assert (
            not self._is_constructed
        ), f"{self.__class__.__name__} already constructed: tried to construct!"

    @abstractmethod
    def on_step(self) -> None:
        assert (
            self._is_constructed
        ), f"{self.__class__.__name__} not constructed: tried to step!"

    @abstractmethod
    def on_reset(self, indices: Indices = None) -> None:
        assert (
            self._is_constructed
        ), f"{self.__class__.__name__} not constructed: tried to reset!"
