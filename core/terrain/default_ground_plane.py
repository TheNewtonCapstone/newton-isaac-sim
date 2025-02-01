from typing import List

import torch

from core.terrain.terrain_builder import BaseTerrainBuilder, BaseTerrainBuild


class DefaultGroundPlaneBuild(BaseTerrainBuild):
    def __init__(
        self,
        path: str,
    ):
        super().__init__(0.0, torch.tensor([]), 0, [0, 0, 0], path)


class DefaultGroundPlaneBuilder(BaseTerrainBuilder):
    def build_from_self(self, position: List[float]) -> DefaultGroundPlaneBuild:
        """
        Notes:
            None of the parameters are used for the default ground plane.
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
        resolution,
        height,
        position,
        path,
    ) -> DefaultGroundPlaneBuild:
        """
        Notes:
            None of the parameters are used for the default ground plane.
        """

        # add a ground plane
        from omni.isaac.core.utils.stage import get_current_stage

        get_current_stage().scene.add_default_ground_plane()

        return DefaultGroundPlaneBuild(path)
