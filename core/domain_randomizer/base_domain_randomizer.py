from abc import ABC, abstractmethod
from typing import Optional

from core.agents import BaseAgent
from core.types import Config, Indices
from core.universe import Universe


class BaseDomainRandomizer(ABC):
    def __init__(
        self,
        seed: int,
        agent: BaseAgent,
        randomizer_settings: Config,
    ):
        self.seed: int = seed

        self._agent: BaseAgent = agent
        self._universe: Optional[Universe] = None

        self.randomizer_settings: Config = randomizer_settings

        self._time: int = 0
        self._is_constructed: bool = False

    @abstractmethod
    def construct(self, universe: Universe) -> None:
        assert (
            not self._is_constructed
        ), f"{self.__class__.__name__} already constructed: tried to construct!"

        self._universe = universe

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
