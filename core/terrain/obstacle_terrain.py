from typing import Optional, List

import numpy as np

import torch
from core.terrain.terrain import BaseTerrainBuild, BaseTerrainBuilder
from torch import Tensor


class ObstacleTerrainBuild(BaseTerrainBuild):
    def __init__(
        self,
        size: float,
        obstacle_height: float,
        obstacle_min_size: float,
        obstacle_max_size: float,
        num_obstacles: int,
        position: List[float],
        path: str,
    ):
        super().__init__(
            size,
            torch.tensor([2, 2], dtype=torch.int32),
            0,
            position,
            path,
        )

        self.obstacle_height = obstacle_height
        self.obstacle_min_size = obstacle_min_size
        self.obstacle_max_size = obstacle_max_size
        self.num_obstacles = num_obstacles


# detail does not affect the flat terrain, the number of vertices is determined by the size
class ObstacleTerrainBuilder(BaseTerrainBuilder):
    def __init__(
        self,
        size: float = None,
        grid_resolution: Tensor = None,
        height: float = 1.0,
        obstacle_height: float = 0.1,
        obstacle_min_size: float = 2.0,
        obstacle_max_size: float = 10.0,
        num_obstacles: int = 30,
        root_path: Optional[str] = None,
    ):
        super().__init__(size, grid_resolution, height, root_path)

        self.obstacle_height = obstacle_height
        self.obstacle_min_size = obstacle_min_size
        self.obstacle_max_size = obstacle_max_size
        self.num_obstacles = num_obstacles

    def build_from_self(self, position: List[float]) -> ObstacleTerrainBuild:
        """
        Notes:
            Resolution and height are not used for flat terrain.
        """

        return self.build(
            self.size,
            self.grid_resolution,
            self.height,
            position,
            self.root_path,
        )

    def build(
        self,
        size,
        grid_resolution,
        height,
        position,
        path,
    ) -> ObstacleTerrainBuild:
        """
        Notes:
            Resolution and height are not used for flat terrain.
        """
        # switch parameters to discrete units
        max_height = self.obstacle_height
        min_size = int(self.obstacle_min_size / 0.25)
        max_size = int(self.obstacle_max_size / 0.25)
        platform_size = int(1.0 / 0.25)

        num_rows = int(size * grid_resolution[0])
        num_cols = int(size * grid_resolution[1])

        heightmap = torch.zeros((num_rows, num_cols))
        (i, j) = heightmap.shape

        height_range = torch.tensor(
            [-max_height, -max_height / 2, max_height / 2, max_height]
        )
        width_range = torch.arange(min_size, max_size, 4)
        length_range = torch.arange(min_size, max_size, 4)

        print(f"{num_rows=}, {num_cols=}, {min_size=}, {max_size=}, {max_height=}")
        print(f"{height_range=}, {width_range=}, {length_range=}")

        for _ in range(self.num_obstacles):
            width = width_range[torch.randint(0, len(width_range), (1,))].item()
            length = length_range[torch.randint(0, len(length_range), (1,))].item()
            start_i = torch.randint(0, i - width, (1,)).item()
            start_j = torch.randint(0, j - length, (1,)).item()
            rand_height = torch.randint(0, len(height_range), (1,))

            heightmap[start_i : start_i + width, start_j : start_j + length] = (
                height_range[rand_height].item()
            )

            print(f"{start_i=}, {start_j=}, {width=}, {length=}, {rand_height=}")

        x1 = int((size - platform_size) // 2)
        x2 = int((size + platform_size) // 2)
        y1 = int((size - platform_size) // 2)
        y2 = int((size + platform_size) // 2)
        heightmap[x1:x2, y1:y2] = 0.0

        terrain_path = BaseTerrainBuilder._add_heightmap_to_world(
            heightmap,
            size,
            height,
            path,
            "obstacle_terrain",
            position,
        )

        from core.utils.physics import set_physics_properties

        set_physics_properties(
            terrain_path,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        return ObstacleTerrainBuild(
            size,
            self.obstacle_height,
            self.obstacle_min_size,
            self.obstacle_max_size,
            self.num_obstacles,
            position,
            terrain_path,
        )
