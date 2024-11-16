from torch import Tensor, tensor

from core.terrain.terrain import TerrainBuilder, TerrainBuild


class DefaultGroundPlaneBuild(TerrainBuild):
    def __init__(
        self,
        path: str,
    ):
        super().__init__(tensor([]), tensor([]), 0, tensor([]), path, None)


class DefaultGroundPlaneBuilder(TerrainBuilder):
    def build_from_self(self, position: Tensor) -> DefaultGroundPlaneBuild:
        """
        Notes:
            None of the parameters are used for the default ground plane.
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
    ) -> DefaultGroundPlaneBuild:
        """
        Notes:
            None of the parameters are used for the default ground plane.
        """

        # add a ground plane
        from omni.isaac.core.utils.stage import get_current_stage

        get_current_stage().scene.add_default_ground_plane()

        return DefaultGroundPlaneBuild(path)
