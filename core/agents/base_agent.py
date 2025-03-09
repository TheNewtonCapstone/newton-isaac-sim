from abc import abstractmethod

from ..base import BaseObject
from ..types import EnvObservations, Actions
from ..universe import Universe


class BaseAgent(BaseObject):
    def __init__(
        self,
        universe: Universe,
    ) -> None:
        super().__init__(universe=universe)

    @abstractmethod
    def step(self, actions: Actions) -> None:
        pass

    @abstractmethod
    def get_observations(self) -> EnvObservations:
        return {}

    @abstractmethod
    def _create(self) -> None:
        pass
