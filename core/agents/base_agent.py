from abc import abstractmethod

from ..base import BaseObject
from ..globals import AGENTS_PATH
from ..types import EnvObservations, Actions
from ..universe import Universe


class BaseAgent(BaseObject):
    def __init__(
        self,
        universe: Universe,
        num_agents: int,
    ) -> None:
        assert num_agents > 0, f"Number of agents must be greater than 0: {num_agents}"

        super().__init__(universe=universe)

        # We type hint universe again here to avoid circular imports
        self._universe: Universe = universe

        self.path: str = AGENTS_PATH
        self.num_agents: int = num_agents

    @abstractmethod
    def construct(self) -> None:
        super().construct()

        from omni.isaac.core.utils.prims import create_prim

        create_prim(
            prim_path=self.path,
            prim_type="Scope",
        )

    @abstractmethod
    def post_construct(self) -> None:
        super().post_construct()

    @abstractmethod
    def step(self, actions: Actions) -> None:
        assert (
            self._is_post_constructed
        ), f"{self.__class__.__name__} not constructed: tried to step!"

    @abstractmethod
    def get_observations(self) -> EnvObservations:
        assert (
            self._is_post_constructed
        ), f"{self.__class__.__name__} not constructed: tried to get observations!"

        return {}
