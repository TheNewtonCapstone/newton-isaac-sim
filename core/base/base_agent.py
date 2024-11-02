from abc import ABC, abstractmethod
import torch


class BaseAgent(ABC):
    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def construct(self, root_path: str, world) -> str:
        pass
