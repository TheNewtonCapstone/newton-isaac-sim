from omni.isaac.core.materials import PhysicsMaterial
from torch import Tensor

import torch
from core.terrain.terrain import TerrainBuild, TerrainBuilder
from perlin_noise import PerlinNoise


class PerlinTerrainBuild(TerrainBuild):
    def __init__(
        self,
        size: Tensor,
        resolution: Tensor,
        height: float,
        position: Tensor,
        path: str,
        physics_mat: PhysicsMaterial,
        octaves: float,
        noise_scale: float,
    ):
        super().__init__(size, resolution, height, position, path, physics_mat)

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
        path="/World/terrains",
        octaves: int = 12,
        noise_scale: float = 4,
    ) -> PerlinTerrainBuild:
        if size is None:
            size = [10, 10]
        if resolution is None:
            resolution = [20, 20]
        if position is None:
            position = [0, 0, 0]

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
            heightmap, size, num_cols, num_rows, height, path, "perlin", position
        )

        physics_material = PhysicsMaterial(terrain_path)

        return PerlinTerrainBuild(
            size,
            resolution,
            height,
            position,
            terrain_path,
            physics_material,
            octaves,
            noise_scale,
        )
