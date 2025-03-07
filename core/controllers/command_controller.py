import random
from typing import Optional, List, Dict

from inputs import GamePad, Keyboard, devices, get_gamepad, get_key

from ..base import BaseObject
from ..logger import Logger
import core.universe
import torch as th

from ..universe import Universe


class CommandController(BaseObject):
    def __init__(self, universe: Universe):
        super().__init__(universe)

        self._universe: Universe = universe

        self._last_action: th.Tensor = th.tensor([0, 0], dtype=th.float32)
        self._receiving_keyboard_commands: bool = False

    @property
    def last_action(self) -> th.Tensor:
        return self._last_action

    @property
    def receiving_keyboard_commands(self) -> bool:
        return self._receiving_keyboard_commands

    @property
    def current_triggers(self) -> List[int]:
        combined_triggers = list(self._current_key_triggers.keys()) + list(
            self._current_gamepad_triggers.keys()
        )
        return [int(trigger) for trigger in combined_triggers]

    @staticmethod
    def get_random_action() -> th.Tensor:
        # Generate random action, with a random direction and magnitude of 1
        random_action = th.rand(2) * 2 - 1

        # Normalize the action
        normalized_action = th.nn.functional.normalize(random_action, p=2, dim=0)

        return normalized_action

    def step(self):
        events = get_gamepad()

        for event in events:
            Logger.info(event)
