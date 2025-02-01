from abc import abstractmethod
from typing import List, Optional

from ..agents import BaseAgent
from ..base import BaseObject
from ..domain_randomizer import BaseDomainRandomizer
from ..terrain import BaseTerrainBuilder, BaseTerrainBuild
from ..terrain.terrain import Terrain
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
        self.num_envs = num_envs

        self.terrain = terrain
        self.terrain_builds: List[BaseTerrainBuild] = []

        self.domain_randomizer: BaseDomainRandomizer = domain_randomizer

    @abstractmethod
    def construct(self) -> None:
        super().construct()

    @abstractmethod
    def post_construct(self) -> None:
        super().post_construct()

    @abstractmethod
    def step(
        self,
        actions: Actions,
    ) -> None:
        assert (
            self._is_post_constructed
        ), f"{self.__class__.__name__} not constructed: tried to step!"

        self._universe.step()

    @abstractmethod
    def reset(self, indices: Optional[Indices] = None) -> EnvObservations:
        assert (
            self._is_post_constructed
        ), f"{self.__class__.__name__} not constructed: tried to reset!"

        return {}

    @abstractmethod
    def get_observations(self) -> EnvObservations:
        assert (
            self._is_post_constructed
        ), f"{self.__class__.__name__} not constructed: tried to get observations!"

        return {}
