import torch
from core.types import IMUData, NoiseFunction


class VecIMU:
    def __init__(
        self,
        path_expr: str,
        local_position: torch.Tensor,
        local_rotation: torch.Tensor,
        noise_function: NoiseFunction,
    ):
        self.path_expr: str = path_expr
        self.local_position: torch.Tensor = local_position
        self.local_rotation: torch.Tensor = local_rotation
        self.num_imus: int = 0

        self._noise_function: NoiseFunction = noise_function
        self._is_constructed: bool = False

    def __del__(self):
        pass

    @property
    def data(self) -> IMUData:
        return self._get_data()

    def construct(self) -> None:
        pass

    def step(self) -> IMUData:
        return self._get_data()

    def reset(self) -> IMUData:
        return self._get_data()

    def _get_data(self) -> IMUData:
        pass
