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

    def get_random_action(self) -> th.Tensor:
        return th.rand(2, device=self.device) * 2 - 1

    def get_random_actions(self, num_actions: int) -> th.Tensor:
        return th.rand((num_actions, 2), device=self.device) * 2 - 1

    def step(self):
        return
        events = get_gamepad()

        for event in events:
            Logger.info(event)
