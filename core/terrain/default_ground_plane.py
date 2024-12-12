from torch import Tensor, tensor

from core.terrain.terrain import BaseTerrainBuilder, BaseTerrainBuild


class DefaultGroundPlaneBuildBase(BaseTerrainBuild):
    def __init__(
        self,
        path: str,
    ):
        super().__init__(tensor([]), tensor([]), 0, tensor([]), path)


class DefaultGroundPlaneBuilderBase(BaseTerrainBuilder):
    def build_from_self(self, position: Tensor) -> DefaultGroundPlaneBuildBase:
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
        path=None,
    ) -> DefaultGroundPlaneBuildBase:
        """
        Notes:
            None of the parameters are used for the default ground plane.
        """

        # add a ground plane
        from omni.isaac.core.utils.stage import get_current_stage

        get_current_stage().scene.add_default_ground_plane()

        return DefaultGroundPlaneBuildBase(path)
