from perlin_noise import PerlinNoise

import torch
from core.terrain.terrain import TerrainBuild, TerrainBuilder
from torch import Tensor


class PerlinTerrainBuild(TerrainBuild):
    def __init__(
        self,
        size: Tensor,
        resolution: Tensor,
        height: float,
        position: Tensor,
        path: str,
        octaves: float,
        noise_scale: float,
    ):
        super().__init__(size, resolution, height, position, path)

        self.octaves = octaves
        self.noise_scale = noise_scale


class PerlinTerrainBuilder(TerrainBuilder):
    def __init__(
        self,
        size: Tensor = None,
        resolution: Tensor = None,
        height: float = 0.05,
        root_path: str = "/Terrains",
        octave: int = 12,
        noise_scale: float = 4,
    ):
        super().__init__(size, resolution, height, root_path)

        self.octaves = octave
        self.noise_scale = noise_scale

    def build_from_self(self, position: Tensor) -> PerlinTerrainBuild:
        return self.build(
            self.size,
            self.resolution,
            self.height,
            position,
            self.root_path,
            self.octaves,
            self.noise_scale,
        )

    @staticmethod
    def build(
        size=None,
        resolution=None,
        height=0.05,
        position=None,
        path=None,
        octaves: int = 12,
        noise_scale: float = 4,
    ) -> PerlinTerrainBuild:
        from core.globals import TERRAINS_PATH

        if size is None:
            size = [10, 10]
        if resolution is None:
            resolution = [20, 20]
        if position is None:
            position = [0, 0, 0]
        if path is None:
            path = TERRAINS_PATH

        num_rows = int(size[0] * resolution[0])
        num_cols = int(size[1] * resolution[1])

        heightmap = torch.zeros((num_rows, num_cols))

        noise = PerlinNoise(octaves=octaves)

        for i in range(num_rows):
            for j in range(num_cols):
                heightmap[i, j] = noise(
                    [i / num_rows * noise_scale, j / num_cols * noise_scale]
                )

        terrain_path = TerrainBuilder._add_heightmap_to_world(
            heightmap,
            size,
            num_cols,
            num_rows,
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
            restitution=1.0,
        )

        return PerlinTerrainBuild(
            size,
            resolution,
            height,
            position,
            terrain_path,
            octaves,
            noise_scale,
        )
