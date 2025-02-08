from typing import Iterable

from carb.input import KeyboardInput, GamepadEvent, GamepadInput

import torch as th


def is_forward_key(key: KeyboardInput) -> bool:
    return key in (KeyboardInput.W, KeyboardInput.UP)


def is_backward_key(key: KeyboardInput) -> bool:
    return key in (KeyboardInput.S, KeyboardInput.DOWN)


def is_left_key(key: KeyboardInput) -> bool:
    return key in (KeyboardInput.A, KeyboardInput.LEFT)


def is_right_key(key: KeyboardInput) -> bool:
    return key in (KeyboardInput.D, KeyboardInput.RIGHT)


def is_right_gamepad(inpt: GamepadInput, right_stick: bool = True) -> bool:
    if inpt == GamepadInput.DPAD_RIGHT:
        return True

    if right_stick:
        return inpt == GamepadInput.RIGHT_STICK_RIGHT
    else:
        return inpt == GamepadInput.LEFT_STICK_RIGHT


def is_left_gamepad(inpt: GamepadInput, right_stick: bool = True) -> bool:
    if inpt == GamepadInput.DPAD_LEFT:
        return True

    if right_stick:
        return inpt == GamepadInput.RIGHT_STICK_LEFT
    else:
        return inpt == GamepadInput.LEFT_STICK_LEFT


def is_up_gamepad(inpt: GamepadInput, right_stick: bool = True) -> bool:
    if inpt == GamepadInput.DPAD_UP:
        return True

    if right_stick:
        return inpt == GamepadInput.RIGHT_STICK_UP
    else:
        return inpt == GamepadInput.LEFT_STICK_UP


def is_down_gamepad(inpt: GamepadInput, right_stick: bool = True) -> bool:
    if inpt == GamepadInput.DPAD_DOWN:
        return True

    if right_stick:
        return inpt == GamepadInput.RIGHT_STICK_DOWN
    else:
        return inpt == GamepadInput.LEFT_STICK_DOWN


def build_movement_keyboard_action(key: KeyboardInput):
    if is_forward_key(key):
        return th.tensor([0, 1], dtype=th.float32)
    elif is_backward_key(key):
        return th.tensor([0, -1], dtype=th.float32)
    elif is_left_key(key):
        return th.tensor([-1, 0], dtype=th.float32)
    elif is_right_key(key):
        return th.tensor([1, 0], dtype=th.float32)
    else:
        return th.tensor([0, 0], dtype=th.float32)


def build_movement_gamepad_action(event: GamepadEvent, right_stick: bool = True):
    abs_val = abs(event.value)

    if is_up_gamepad(event.input, right_stick):
        return th.tensor([0, 1], dtype=th.float32) * abs_val
    elif is_down_gamepad(event.input, right_stick):
        return th.tensor([0, -1], dtype=th.float32) * abs_val
    elif is_left_gamepad(event.input, right_stick):
        return th.tensor([-1, 0], dtype=th.float32) * abs_val
    elif is_right_gamepad(event.input, right_stick):
        return th.tensor([1, 0], dtype=th.float32) * abs_val
    else:
        return th.tensor([0, 0], dtype=th.float32)


def combine_movement_inputs(inputs: Iterable[th.Tensor]) -> th.Tensor:
    sum_inputs = th.tensor([0, 0], dtype=th.float32)

    for inpt in inputs:
        sum_inputs += inpt

    if abs(sum_inputs[0]) > 1:
        sum_inputs[0] = 1 if sum_inputs[0] > 0 else -1
    if abs(sum_inputs[1]) > 1:
        sum_inputs[1] = 1 if sum_inputs[1] > 0 else -1

    return sum_inputs
