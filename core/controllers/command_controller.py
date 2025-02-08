import random
from typing import Optional, List, Dict

from carb.input import (
    Keyboard,
    Gamepad,
    acquire_input_interface,
    GamepadEvent,
    KeyboardEvent,
    KeyboardInput,
    KeyboardEventType,
    GamepadInput,
)

import omni.appwindow
from ..base import BaseObject
from ..logger import Logger
import core.universe
from omni.appwindow import IAppWindow
import torch as th


class CommandController(BaseObject):
    def __init__(self, universe: core.universe.Universe):
        super().__init__(universe)

        self._universe: core.universe.Universe = universe

        self._app_window: Optional[IAppWindow] = None

        self._gamepad: Optional[Gamepad] = None
        self._keyboard: Optional[Keyboard] = None

        self._keyboard_event_sub: Optional[int] = None
        self._gamepad_event_sub: Optional[int] = None

        self._current_key_triggers: Dict[KeyboardInput, th.Tensor] = {}
        self._current_gamepad_triggers: Dict[GamepadInput, th.Tensor] = {}
        self._last_action: th.Tensor = th.tensor([0, 0], dtype=th.float32)
        self._receiving_keyboard_commands: bool = False

    def __del__(self):
        # Unsubscribe from events
        self._keyboard_event_sub = None
        self._gamepad_event_sub = None

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
        random_x = random.random() * 2 - 1
        random_y = random.random() * 2 - 1

        return th.tensor([random_x, random_y], dtype=th.float32)

    def construct(self) -> None:
        super().construct()

        self._app_window = omni.appwindow.get_default_app_window()

        # first connected gamepad and keyboard
        self._gamepad = self._app_window.get_gamepad(0)
        self._keyboard = self._app_window.get_keyboard()

        input_iface = acquire_input_interface()
        self._keyboard_event_sub = input_iface.subscribe_to_keyboard_events(
            self._keyboard,
            self._keyboard_event_cb,
        )
        self._gamepad_event_sub = input_iface.subscribe_to_gamepad_events(
            self._gamepad,
            self._gamepad_event_cb,
        )

        self._is_constructed = True

    def post_construct(self) -> None:
        super().post_construct()

        self._is_post_constructed = True

    def _keyboard_event_cb(self, event: KeyboardEvent) -> bool:
        if event.type == KeyboardEventType.CHAR:
            return False

        # enables/disables receiving keyboard commands (avoids conflicts with other parts of the system)
        if (
            event.input == KeyboardInput.TAB
            and event.type == KeyboardEventType.KEY_RELEASE
        ):
            self._receiving_keyboard_commands = not self._receiving_keyboard_commands

            Logger.info(
                f"Toggled receiving keyboard commands: {self._receiving_keyboard_commands}"
            )
            return False

        if not self._receiving_keyboard_commands:
            return False

        # ignore key release events
        if event.type == KeyboardEventType.KEY_RELEASE:
            if event.input in self._current_key_triggers:
                del self._current_key_triggers[event.input]

                # ensure that other keys are still processed
                self._process_actions()

            return False

        from core.utils.input import build_movement_keyboard_action

        movement_action = build_movement_keyboard_action(event.input)

        self._current_key_triggers[event.input] = movement_action
        self._process_actions()

        return True

    def _gamepad_event_cb(self, event: GamepadEvent) -> bool:
        from core.utils.input import build_movement_gamepad_action

        movement_action = build_movement_gamepad_action(event, right_stick=True)

        self._current_gamepad_triggers[event.input] = movement_action
        self._process_actions()

        return True

    def _process_actions(self) -> None:
        from core.utils.input import combine_movement_inputs

        all_actions = list(self._current_key_triggers.values()) + list(
            self._current_gamepad_triggers.values()
        )

        self._last_action = combine_movement_inputs(all_actions)
