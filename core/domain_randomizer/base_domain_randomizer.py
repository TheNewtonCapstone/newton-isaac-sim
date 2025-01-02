from abc import abstractmethod
from typing import Optional

from ..agents import BaseAgent
from ..base import BaseObject
from ..types import Config, Indices
from ..universe import Universe


class BaseDomainRandomizer(BaseObject):
    def __init__(
        self,
        universe: Universe,
        seed: int,
        agent: BaseAgent,
        randomizer_settings: Config,
    ):
        super().__init__(universe=universe)

        self.seed: int = seed

        self._agent: BaseAgent = agent

        self.randomizer_settings: Config = randomizer_settings

        self._time: int = 0

    @abstractmethod
    def construct(self) -> None:
        super().construct()

    @abstractmethod
    def post_construct(self) -> None:
        super().post_construct()

    @abstractmethod
    def on_step(self) -> None:
        assert (
            self._is_post_constructed
        ), f"{self.__class__.__name__} not constructed: tried to step!"

    @abstractmethod
    def on_reset(self, indices: Indices = None) -> None:
        assert (
            self._is_post_constructed
        ), f"{self.__class__.__name__} not constructed: tried to reset!"
