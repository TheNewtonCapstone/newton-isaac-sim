import numpy as np


class SubTerrain:
    def __init__(
        self,
        width: int = 256,
        length: int = 256,
        vertical_resolution: float = 1.0,
        horizontal_resolution: float = 1.0,
    ):
        self.width: int = width
        self.length: int = length

        self.vertical_resolution: float = vertical_resolution
        self.horizontal_resolution: float = horizontal_resolution

        self.height_field: np.ndarray = np.zeros(
            (self.width, self.length),
            dtype=np.int16,
        )
