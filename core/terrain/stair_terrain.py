from typing import List

import numpy as np
from torch import Tensor

from core.terrain.terrain_builder import BaseTerrainBuild, BaseTerrainBuilder


class StairsTerrainBuild(BaseTerrainBuild):
    def __init__(
        self,
        size: float,
        grid_resolution: Tensor,
        height: float,
        position: List[float],
        path: str,
        stair_height: float,
        number_of_steps: int,
    ):
        super().__init__(size, grid_resolution, height, position, path)

        self.stair_height = stair_height
        self.number_of_steps = number_of_steps


class StairsTerrainBuilder(BaseTerrainBuilder):
    def __init__(
        self,
        size: float = None,
        grid_resolution: Tensor = None,
        height: float = 1.0,
        root_path: str = "/Terrains",
        stair_height: float = 0.1,
        number_of_steps: int = 10,
    ):
        super().__init__(size, grid_resolution, height, root_path)

        self.stair_height = stair_height
        self.number_of_steps = number_of_steps

    def build_from_self(self, position: List[float]) -> StairsTerrainBuild:
        return self.build(
            self.size,
            self.grid_resolution,
            self.height,
            position,
            self.root_path,
            self.stair_height,
            self.number_of_steps,
        )

    @staticmethod
    def build(
        size,
        grid_resolution,
        height,
        position,
        path,
        step_height: float = 0.1,
        number_of_steps: int = 10,
    ) -> StairsTerrainBuild:
        num_rows = int(size * grid_resolution[0])
        num_cols = int(size * grid_resolution[1])

        heightmap = np.zeros((num_rows, num_cols))

        # Generate pyramid-like terrain with steps (stairs)
        step_width = num_rows / (number_of_steps * 2)

        start_height = 0
        start_x = 0
        start_y = 0
        stop_x = num_rows
        stop_y = num_cols

        while (stop_x > start_x) and (stop_y > start_y):
            start_x = int(start_x + step_width)
            stop_x = int(stop_x - step_width)
            start_y = int(start_y + step_width)
            stop_y = int(stop_y - step_width)
            start_height += step_height
            heightmap[start_x:stop_x, start_y:stop_y] = start_height

        # Generate pyramid-like terrain with steps (stairs)
        terrain_path = BaseTerrainBuilder._add_heightmap_to_world(
            heightmap,
            size,
            height,
            path,
            "ascending_stairs",
            position,
        )

        from core.utils.physics import set_physics_properties

        set_physics_properties(
            terrain_path,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        return StairsTerrainBuild(
            size,
            grid_resolution,
            start_height,
            position,
            terrain_path,
            step_height,
            number_of_steps,
        )
