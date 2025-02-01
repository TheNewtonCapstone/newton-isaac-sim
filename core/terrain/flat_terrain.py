from typing import Optional, List

import numpy as np

import torch
from core.terrain.terrain_builder import BaseTerrainBuild, BaseTerrainBuilder
from torch import Tensor


class FlatTerrainBuild(BaseTerrainBuild):
    def __init__(
        self,
        size: float,
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


# detail does not affect the flat terrain, the number of vertices is determined by the size
class FlatTerrainBuilder(BaseTerrainBuilder):
    def __init__(
        self,
        size: float = None,
        grid_resolution: Tensor = None,
        height: float = 0,
        root_path: Optional[str] = None,
    ):
        super().__init__(size, grid_resolution, height, root_path)

    def build_from_self(self, position: List[float]) -> FlatTerrainBuild:
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
    ) -> FlatTerrainBuild:
        """
        Notes:
            Resolution and height are not used for flat terrain.
        """
        num_rows = int(size * grid_resolution[0])
        num_cols = int(size * grid_resolution[1])

        heightmap = np.zeros((num_rows, num_cols))

        terrain_path = BaseTerrainBuilder._add_heightmap_to_world(
            heightmap,
            size,
            height,
            path,
            "flat",
            position,
        )

        from core.utils.physics import set_physics_properties

        set_physics_properties(
            terrain_path,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        return FlatTerrainBuild(
            size,
            position,
            terrain_path,
        )
