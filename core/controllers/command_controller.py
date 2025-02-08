import random
from typing import Optional, List, Dict, Iterable

from core.base import BaseObject
from core.logger import Logger
from core.universe import Universe

import omni.appwindow
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

from omni.appwindow import IAppWindow

from pxr import Gf


class CommandController(BaseObject):
    def __init__(self, universe: Universe):
        super().__init__(universe)

        self._universe: Universe = universe

        self._app_window: IAppWindow = omni.appwindow.get_default_app_window()

        # first connected gamepad and keyboard
        self._gamepad: Gamepad = self._app_window.get_gamepad(0)
        self._keyboard: Keyboard = self._app_window.get_keyboard()

        input_iface = acquire_input_interface()
        self._keyboard_event_sub: Optional[int] = (
            input_iface.subscribe_to_keyboard_events(
                self._keyboard,
                self._keyboard_event_cb,
            )
        )
        self._gamepad_event_sub: Optional[int] = (
            input_iface.subscribe_to_gamepad_events(
                self._gamepad,
                self._gamepad_event_cb,
            )
        )

        self._current_key_actions: Dict[KeyboardInput | GamepadInput, Gf.Vec2f] = {}
        self._last_action: Gf.Vec2f = Gf.Vec2f(0, 0)
        self._receiving_keyboard_commands: bool = False

    def __del__(self):
        # Unsubscribe from events
        self._keyboard_event_sub = None
        self._gamepad_event_sub = None

    @property
    def last_action(self) -> Gf.Vec2f:
        return self._last_action

    @property
    def receiving_keyboard_commands(self) -> bool:
        return self._receiving_keyboard_commands

    @staticmethod
    def get_random_action() -> Gf.Vec2f:
        random_x = random.random() * 2 - 1
        random_y = random.random() * 2 - 1

        return Gf.Vec2f(random_x, random_y)

    def construct(self) -> None:
        super().construct()

    def post_construct(self) -> None:
        super().post_construct()

    def _keyboard_event_cb(self, event: KeyboardEvent) -> bool:
        if event.type == KeyboardEventType.CHAR:
            return False

        if (
            event.input == KeyboardInput.TAB
            and event.type == KeyboardEventType.KEY_RELEASE
        ):
            self._receiving_keyboard_commands = not self._receiving_keyboard_commands

            Logger.info(
                f"Toggled receiving keyboard commands: {self._receiving_keyboard_commands}"
            )
            return False

        from core.utils.input import build_movement_keyboard_action

        if event.type == KeyboardEventType.KEY_RELEASE:
            if event.input in self._current_key_actions:
                del self._current_key_actions[event.input]

                # ensure that other keys are still processed
                self._last_action = self._process_actions()

                Logger.error(
                    f"KR: Last action: {self._last_action}, current key actions: {self._current_key_actions}, event: {event}"
                )
            return False

        movement_action = build_movement_keyboard_action(event.input)

        self._current_key_actions[event.input] = movement_action
        self._last_action = self._process_actions()

        Logger.error(
            f"K: Last action: {self._last_action}, current key actions: {self._current_key_actions}, event: {event}"
        )

        return True

    def _gamepad_event_cb(self, event: GamepadEvent) -> bool:
        from core.utils.input import build_movement_gamepad_action

        movement_action = build_movement_gamepad_action(event)

        self._last_action = self._process_actions(movement_action)

        Logger.error(
            f"G: Last action: {self._last_action}, current key actions: {self._current_key_actions}, event: {event}"
        )

        return True

    def _process_actions(
        self,
        gamepad_action: Optional[List[Gf.Vec2f]] = None,
    ) -> Gf.Vec2f:
        if gamepad_action is None:
            gamepad_action = []

        from core.utils.input import combine_movement_inputs

        all_actions = list(self._current_key_actions.values()) + gamepad_action

        combined_input = combine_movement_inputs(all_actions)

        return combined_input
