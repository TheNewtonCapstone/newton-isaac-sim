from core.terrain.terrain import TerrainBuild, TerrainBuilder
import torch


class FlatTerrainBuild(TerrainBuild):
    def __init__(
        self,
        stage,
        size: list[int],
        position: list[float],
        path: str,
    ):
        super().__init__(stage, size, [2, 2], 0, position, path)


# detail does not affect the flat terrain, the number of vertices is determined by the size
class FlatTerrainBuilder(TerrainBuilder):
    def __init__(
        self,
        size: list[int] = None,
        resolution: list[int] = None,
        height: float = 0,
        root_path: str = None,
    ):
        super().__init__(size, resolution, height, root_path)

    def build_from_self(self, stage, position: list[float]) -> FlatTerrainBuild:
        """
        Notes:
            Resolution and height are not used for flat terrain.
        """

        return self.build(
            stage, self.size, self.resolution, self.height, position, self.base_path
        )

    @staticmethod
    def build(
        stage,
        size=None,
        resolution=None,
        height=0,
        position=None,
        path="/World/terrains",
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

        from core.utils.physics import set_physics_properties

        set_physics_properties(
            terrain_path, static_friction=1, dynamic_friction=1, restitution=0
        )

        return FlatTerrainBuild(
            stage,
            size,
            position,
            terrain_path,
        )
