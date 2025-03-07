from abc import abstractmethod
from typing import Optional

from ..agents import BaseAgent
from ..base import BaseObject
from ..domain_randomizer import BaseDomainRandomizer
from ..terrain import Terrain
from ..types import EnvObservations, Actions, Indices
from ..universe import Universe


class BaseEnv(BaseObject):
    def __init__(
        self,
        universe: Universe,
        agent: BaseAgent,
        num_envs: int,
        terrain: Terrain,
        domain_randomizer: BaseDomainRandomizer,
    ) -> None:
        super().__init__(
            universe=universe,
        )

        # We type hint universe again here to avoid circular imports
        self._universe: Universe = universe

        self.agent: BaseAgent = agent
        self.num_envs: int = num_envs

        self.terrain: Terrain = terrain

        self.domain_randomizer: BaseDomainRandomizer = domain_randomizer

    @abstractmethod
    def pre_build(self) -> None:
        super().pre_build()

    @abstractmethod
    def post_build(self) -> None:
        super().post_build()

    @abstractmethod
    def step(
        self,
        actions: Actions,
        render: bool = True,
    ) -> None:
        assert self.is_built, f"{self.__class__.__name__} not built: tried to step!"

        self._universe.step(render=render)

    @abstractmethod
    def reset(self, indices: Optional[Indices] = None) -> EnvObservations:
        assert self.is_built, f"{self.__class__.__name__} not built: tried to reset!"

        return {}

    @abstractmethod
    def get_observations(self) -> EnvObservations:
        assert (
            self.is_built
        ), f"{self.__class__.__name__} not built: tried to get observations!"

        return {}
