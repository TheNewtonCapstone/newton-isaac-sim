from typing import List

import numpy as np
import torch
from perlin_noise import PerlinNoise

from core.terrain.terrain import BaseTerrainBuild, BaseTerrainBuilder


class PerlinTerrainBuild(BaseTerrainBuild):
    def __init__(
        self,
        size: float,
        resolution: torch.Tensor,
        height: float,
        position: List[float],
        path: str,
        octaves: float,
        noise_scale: float,
    ):
        super().__init__(size, resolution, height, position, path)

        self.octaves = octaves
        self.noise_scale = noise_scale


class PerlinTerrainBuilder(BaseTerrainBuilder):
    def __init__(
        self,
        size: float = None,
        grid_resolution: torch.Tensor = None,
        height: float = 0.05,
        root_path: str = "/Terrains",
        octave: int = 12,
        noise_scale: float = 4,
    ):
        super().__init__(size, grid_resolution, height, root_path)

        self.octaves = octave
        self.noise_scale = noise_scale

    def build_from_self(self, position: List[float]) -> PerlinTerrainBuild:
        return self.build(
            self.size,
            self.grid_resolution,
            self.height,
            position,
            self.root_path,
            self.octaves,
            self.noise_scale,
        )

    @staticmethod
    def build(
        size,
        grid_resolution,
        height,
        position,
        path,
        octaves: int = 12,
        noise_scale: float = 4,
    ) -> PerlinTerrainBuild:
        from core.globals import TERRAINS_PATH

        if size is None:
            size = 10.0
        if grid_resolution is None:
            grid_resolution = [20, 20]
        if position is None:
            position = [0, 0, 0]
        if path is None:
            path = TERRAINS_PATH

        num_rows = int(size * grid_resolution[0])
        num_cols = int(size * grid_resolution[1])

        heightmap = np.zeros((num_rows, num_cols))

        noise = PerlinNoise(octaves=octaves)

        for i in range(num_rows):
            for j in range(num_cols):
                heightmap[i, j] = noise(
                    [i / num_rows * noise_scale, j / num_cols * noise_scale]
                )

        terrain_path = BaseTerrainBuilder._add_heightmap_to_world(
            heightmap,
            size,
            height,
            path,
            "perlin",
            position,
        )

        from core.utils.physics import set_physics_properties

        set_physics_properties(
            terrain_path,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        return PerlinTerrainBuild(
            size,
            grid_resolution,
            height,
            position,
            terrain_path,
            octaves,
            noise_scale,
        )
