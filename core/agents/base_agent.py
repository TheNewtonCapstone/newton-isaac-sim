from abc import abstractmethod

from ..base import BaseObject
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

        self.num_agents: int = num_agents

    @abstractmethod
    def step(self, actions: Actions) -> None:
        pass

    @abstractmethod
    def get_observations(self) -> EnvObservations:
        return {}

    @abstractmethod
    def _create(self) -> None:
        pass
