from abc import abstractmethod, ABC
from typing import Optional

import torch
from torch import Tensor


class BaseTerrainBuild:
    def __init__(
        self,
        size: Tensor,
        resolution: Tensor,
        height: float,
        position: Tensor,
        path: str,
    ):
        self.size = size
        self.position = position
        self.resolution = resolution
        self.height = height

        self.path = path


class BaseTerrainBuilder(ABC):
    def __init__(
        self,
        size: Tensor = None,
        resolution: Tensor = None,
        height: float = 1,
        root_path: Optional[str] = None,
    ):
        from core.globals import TERRAINS_PATH

        if size is None:
            size = torch.tensor([10, 10])
        if resolution is None:
            resolution = torch.tensor([20, 20])
        if root_path is None:
            root_path = TERRAINS_PATH

        self.size = size
        self.resolution = resolution
        self.height = height
        self.root_path = root_path

        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.prims import is_prim_path_valid

        if not is_prim_path_valid(root_path):
            create_prim(
                prim_path=root_path,
                prim_type="Scope",
            )

    def build_from_self(self, position: Tensor) -> BaseTerrainBuild:
        return self.build(
            self.size,
            self.resolution,
            self.height,
            position,
            self.root_path,
        )

    @abstractmethod
    def build(
        self,
        size: Optional[Tensor] = None,
        resolution: Optional[Tensor] = None,
        height: float = 1,
        position: Optional[Tensor] = None,
        path: Optional[str] = None,
    ) -> BaseTerrainBuild:
        """
        Builds a terrain in the stage, according to the class's implementation.

        Args:
            size: Size of the terrain in the stage's units.
            resolution: Number of vertices per terrain.
            height: Height of the terrain in the stage's units.
            position: Position of the terrain in the stage's units.
            path: Path to the terrain in the stage.
        """
        pass

    @staticmethod
    def _add_heightmap_to_world(
        heightmap: Tensor,
        size: Tensor,
        num_cols: int,
        num_rows: int,
        height: float,
        base_path: str,
        builder_name: str,
        position: Tensor,
    ) -> str:
        vertices, triangles = BaseTerrainBuilder._heightmap_to_mesh(
            heightmap, size, num_cols, num_rows, height
        )

        return BaseTerrainBuilder._add_mesh_to_world(
            vertices, triangles, base_path, builder_name, size, position
        )

    @staticmethod
    def _heightmap_to_mesh(
        heightmap: Tensor,
        size: Tensor,
        num_cols: int,
        num_rows: int,
        height: float,
    ) -> tuple[Tensor, Tensor]:
        # from https://github.com/isaac-sim/OmniIsaacGymEnvs/blob/main/omniisaacgymenvs/utils/terrain_utils/terrain_utils.py

        x = torch.linspace(0, (size[0]), num_cols)
        y = torch.linspace(0, (size[1]), num_rows)
        xx, yy = torch.meshgrid(x, y)

        vertices = torch.zeros((num_cols * num_rows, 3), dtype=torch.float32)
        vertices[:, 0] = xx.flatten()
        vertices[:, 1] = yy.flatten()
        vertices[:, 2] = heightmap.flatten() * height
        triangles = torch.ones(
            (2 * (num_rows - 1) * (num_cols - 1), 3), dtype=torch.int32
        )
        for i in range(num_cols - 1):
            # indices for the 4 vertices of the square
            ind0 = torch.arange(0, num_rows - 1) + i * num_rows
            ind1 = ind0 + 1
            ind2 = ind0 + num_rows
            ind3 = ind2 + 1

            # there are 2 triangles per square
            # and self.size[1] - 1 squares per col
            start = i * (num_rows - 1) * 2
            end = start + (num_rows - 1) * 2

            # first set of triangles (top left)
            triangles[start:end:2, 0] = ind0
            triangles[start:end:2, 1] = ind3
            triangles[start:end:2, 2] = ind1

            # second set of triangles (bottom right)
            triangles[start + 1 : end : 2, 0] = ind0
            triangles[start + 1 : end : 2, 1] = ind2
            triangles[start + 1 : end : 2, 2] = ind3

        return vertices, triangles

    @staticmethod
    def _add_mesh_to_world(
        vertices: Tensor,
        triangles: Tensor,
        base_path: str,
        builder_name: str,
        size: Tensor,
        position: Tensor,
    ) -> str:
        from core.utils.usd import find_matching_prims
        from omni.isaac.core.utils.prims import (
            define_prim,
            create_prim,
            is_prim_path_valid,
        )
        from pxr import UsdPhysics, PhysxSchema
        import numpy as np

        # generate an informative and unique name from the type of builder
        builder_group_prim_path = f"{base_path}/{builder_name}"
        prim_path_expr = f"{builder_group_prim_path}/terrain_.*"
        num_of_existing_terrains = len(find_matching_prims(prim_path_expr))
        prim_path = f"{base_path}/{builder_name}/terrain_{num_of_existing_terrains}"
        num_faces = triangles.shape[0]

        if not is_prim_path_valid(builder_group_prim_path):
            define_prim(
                builder_group_prim_path,
                prim_type="Scope",
            )

        centered_position = [
            position[0] - size[0] / 2,
            position[1] - size[1] / 2,
            position[2],
        ]

        # creates the terrain's root prim
        create_prim(
            prim_path,
            prim_type="Xform",
            position=np.asarray(centered_position),
        )

        # creates the mesh prim, that actually collides
        mesh_prim = create_prim(
            prim_path + "/mesh",
            prim_type="Mesh",
            scale=[1, 1, 1],
            attributes={
                "points": vertices.numpy(),
                "faceVertexIndices": triangles.flatten().numpy(),
                "faceVertexCounts": np.asarray([3] * num_faces),
                "subdivisionScheme": "bilinear",
            },
        )

        # ensure that we have all the necessary APIs
        collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
        collision_api.CreateCollisionEnabledAttr(True)

        return prim_path
