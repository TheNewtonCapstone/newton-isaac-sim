from typing import Iterable

from carb.input import KeyboardInput, GamepadEvent, GamepadInput

from pxr import Gf


def is_forward_key(key: KeyboardInput) -> bool:
    return key in (KeyboardInput.W, KeyboardInput.UP)


def is_backward_key(key: KeyboardInput) -> bool:
    return key in (KeyboardInput.S, KeyboardInput.DOWN)


def is_left_key(key: KeyboardInput) -> bool:
    return key in (KeyboardInput.A, KeyboardInput.LEFT)


def is_right_key(key: KeyboardInput) -> bool:
    return key in (KeyboardInput.D, KeyboardInput.RIGHT)


def is_right_gamepad(inpt: GamepadInput, right_stick: bool = False) -> bool:
    if inpt == GamepadInput.DPAD_RIGHT:
        return True

    if right_stick:
        return inpt == GamepadInput.RIGHT_STICK_RIGHT
    else:
        return inpt == GamepadInput.LEFT_STICK_RIGHT


def is_left_gamepad(inpt: GamepadInput, right_stick: bool = False) -> bool:
    if inpt == GamepadInput.DPAD_LEFT:
        return True

    if right_stick:
        return inpt == GamepadInput.RIGHT_STICK_LEFT
    else:
        return inpt == GamepadInput.LEFT_STICK_LEFT


def is_up_gamepad(inpt: GamepadInput, right_stick: bool = False) -> bool:
    if inpt == GamepadInput.DPAD_UP:
        return True

    if right_stick:
        return inpt == GamepadInput.RIGHT_STICK_UP
    else:
        return inpt == GamepadInput.LEFT_STICK_UP


def is_down_gamepad(inpt: GamepadInput, right_stick: bool = False) -> bool:
    if inpt == GamepadInput.DPAD_DOWN:
        return True

    if right_stick:
        return inpt == GamepadInput.RIGHT_STICK_DOWN
    else:
        return inpt == GamepadInput.LEFT_STICK_DOWN


def build_movement_keyboard_action(key: KeyboardInput):
    if is_forward_key(key):
        return Gf.Vec2f(0, 1)
    elif is_backward_key(key):
        return Gf.Vec2f(0, -1)
    elif is_left_key(key):
        return Gf.Vec2f(-1, 0)
    elif is_right_key(key):
        return Gf.Vec2f(1, 0)
    else:
        return Gf.Vec2f(0, 0)


def build_movement_gamepad_action(event: GamepadEvent):
    abs_val = abs(event.value)

    if is_up_gamepad(event.input):
        return Gf.Vec2f(0, 1) * abs_val
    elif is_down_gamepad(event.input):
        return Gf.Vec2f(0, -1) * abs_val
    elif is_left_gamepad(event.input):
        return Gf.Vec2f(-1, 0) * abs_val
    elif is_right_gamepad(event.input):
        return Gf.Vec2f(1, 0) * abs_val
    else:
        return Gf.Vec2f(0, 0)


def combine_movement_inputs(inputs: Iterable[Gf.Vec2f]) -> Gf.Vec2f:
    sum_inputs = Gf.Vec2f(0, 0)

    for inpt in inputs:
        sum_inputs += inpt

    if abs(sum_inputs[0]) > 1:
        sum_inputs[0] = 1 if sum_inputs[0] > 0 else -1
    if abs(sum_inputs[1]) > 1:
        sum_inputs[1] = 1 if sum_inputs[1] > 0 else -1

    return sum_inputs
