import torch
from core.terrain.terrain import BaseTerrainBuild, BaseTerrainBuilder
from torch import Tensor


class AscendingStairsTerrainBuild(BaseTerrainBuild):
    def __init__(
        self,
        size: Tensor,
        resolution: Tensor,
        height: float,
        position: Tensor,
        path: str,
        stair_height: float,
        number_of_steps: int,
    ):
        super().__init__(size, resolution, height, position, path)

        self.stair_height = stair_height
        self.number_of_steps = number_of_steps


class AscendingStairsTerrainBuilder(BaseTerrainBuilder):
    def __init__(
        self,
        size: Tensor = None,
        resolution: Tensor = None,
        height: float = 0.05,
        root_path: str = "/Terrains",
        stair_height: float = 0.2,
        number_of_steps: int = 5,
    ):
        super().__init__(size, resolution, height, root_path)

        self.stair_height = stair_height
        self.number_of_steps = number_of_steps

    def build_from_self(self, position: Tensor) -> AscendingStairsTerrainBuild:
        return self.build(
            self.size,
            self.resolution,
            self.height,
            position,
            self.root_path,
            self.stair_height,
            self.number_of_steps,
        )

    @staticmethod
    def build(
        size=None,
        resolution=None,
        height=1,
        position=None,
        path=None,
        stair_height: float = 0.20,
        number_of_steps: int = 5,
    ) -> AscendingStairsTerrainBuild:
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

        # Generate pyramid-like terrain with steps (stairs)
        step_width = num_rows / number_of_steps

        platform_size = 1.0
        start_height = 0
        start_x = 0
        start_y = 0
        stop_x = size[0]
        stop_y = size[1]

        while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
            start_x = int(start_x + step_width)
            stop_x = int(stop_x - step_width)
            start_y = int(start_y + step_width)
            stop_y = int(stop_y - step_width)
            start_height += stair_height
            heightmap[start_x:stop_x, start_y:stop_y] = start_height

        # Generate pyramid-like terrain with steps (stairs)
        terrain_path = BaseTerrainBuilder._add_heightmap_to_world(
            heightmap,
            size,
            num_cols,
            num_rows,
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

        return AscendingStairsTerrainBuild(
            size,
            resolution,
            height,
            position,
            terrain_path,
            stair_height,
            number_of_steps,
        )
