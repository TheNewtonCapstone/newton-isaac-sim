from abc import abstractmethod

from genesis.engine.entities import RigidEntity

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

    def pre_build(self) -> None:
        super().pre_build()

        self._is_pre_built = True

    def post_build(self) -> None:
        super().post_build()

        self._is_post_built = True

    @abstractmethod
    def on_step(self) -> None:
        assert self.is_built, f"{self.__class__.__name__} not built: tried to step!"

    @abstractmethod
    def on_reset(self, indices: Indices = None) -> None:
        assert self.is_built, f"{self.__class__.__name__} not built: tried to reset!"
