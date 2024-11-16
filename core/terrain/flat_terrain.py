from omni.isaac.core.materials import PhysicsMaterial
from torch import Tensor

import torch
from core.terrain.terrain import TerrainBuild, TerrainBuilder


class FlatTerrainBuild(TerrainBuild):
    def __init__(
        self,
        size: Tensor,
        position: Tensor,
        path: str,
        physics_mat: PhysicsMaterial,
    ):
        super().__init__(
            size,
            torch.tensor([2, 2], dtype=torch.int32),
            0,
            position,
            path,
            physics_mat,
        )


# detail does not affect the flat terrain, the number of vertices is determined by the size
class FlatTerrainBuilder(TerrainBuilder):
    def __init__(
        self,
        size: Tensor = None,
        resolution: Tensor = None,
        height: float = 0,
        root_path: str = "/Terrains",
    ):
        super().__init__(size, resolution, height, root_path)

    def build_from_self(self, position: Tensor) -> FlatTerrainBuild:
        """
        Notes:
            Resolution and height are not used for flat terrain.
        """

        return self.build(
            self.size,
            self.resolution,
            self.height,
            position,
            self.root_path,
        )

    def build(
        self,
        size=None,
        resolution=None,
        height=0,
        position=None,
        path="/Terrains",
    ) -> FlatTerrainBuild:
        """
        Notes:
            Resolution and height are not used for flat terrain.
        """

        if size is None:
            size = [20, 20]
        if position is None:
            position = [0, 0, 0]

        heightmap = torch.tensor([[0.0] * 2] * 2)

        terrain_path = TerrainBuilder._add_heightmap_to_world(
            heightmap, size, 2, 2, height, path, "flat", position
        )

        physics_mat = PhysicsMaterial(terrain_path)

        return FlatTerrainBuild(
            size,
            position,
            terrain_path,
            physics_mat,
        )
