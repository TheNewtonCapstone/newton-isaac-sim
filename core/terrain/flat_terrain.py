from typing import Optional

import torch
from core.terrain.terrain import BaseTerrainBuild, BaseTerrainBuilder
from torch import Tensor


class FlatBaseTerrainBuild(BaseTerrainBuild):
    def __init__(
        self,
        size: Tensor,
        position: Tensor,
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
class FlatBaseTerrainBuilder(BaseTerrainBuilder):
    def __init__(
        self,
        size: Tensor = None,
        resolution: Tensor = None,
        height: float = 0,
        root_path: Optional[str] = None,
    ):
        super().__init__(size, resolution, height, root_path)

    def build_from_self(self, position: Tensor) -> FlatBaseTerrainBuild:
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
        path=None,
    ) -> FlatBaseTerrainBuild:
        """
        Notes:
            Resolution and height are not used for flat terrain.
        """

        from core.globals import TERRAINS_PATH

        if size is None:
            size = [20, 20]
        if position is None:
            position = [0, 0, 0]
        if path is None:
            path = TERRAINS_PATH

        heightmap = torch.tensor([[0.0] * 2] * 2)

        terrain_path = BaseTerrainBuilder._add_heightmap_to_world(
            heightmap,
            size,
            2,
            2,
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

        return FlatBaseTerrainBuild(
            size,
            position,
            terrain_path,
        )
