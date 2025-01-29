from abc import abstractmethod, ABC
from typing import Optional, Tuple, List

import numpy as np

import torch
from torch import Tensor


class BaseTerrainBuild:
    def __init__(
        self,
        size: float,
        resolution: Tensor,
        height: float,
        position: List[float],
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
        size: float = None,
        resolution: Tensor = None,
        height: float = 1,
        root_path: Optional[str] = None,
    ):
        from core.globals import TERRAINS_PATH

        if size is None:
            size = 10.0
        if resolution is None:
            resolution = torch.tensor([48, 48])
        if root_path is None:
            root_path = TERRAINS_PATH

        self.size = size
        self.grid_resolution = resolution
        self.height = height
        self.root_path = root_path

        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.prims import is_prim_path_valid

        if not is_prim_path_valid(root_path):
            create_prim(
                prim_path=root_path,
                prim_type="Scope",
            )

    def build_from_self(self, position: List[float]) -> BaseTerrainBuild:
        return self.build(
            self.size,
            self.grid_resolution,
            self.height,
            position,
            self.root_path,
        )

    @abstractmethod
    def build(
        self,
        size: Optional[float],
        resolution: Optional[Tensor],
        height: float,
        position: Optional[List[float]],
        path: Optional[str],
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
        heightmap: np.ndarray,
        scale: float,
        height: float,
        base_path: str,
        builder_name: str,
        position: List[float],
    ) -> str:
        vertices, triangles = BaseTerrainBuilder._heightmap_to_mesh(
            heightmap,
            height,
            scale / heightmap.shape[0],
        )

        return BaseTerrainBuilder._add_mesh_to_world(
            vertices,
            triangles,
            base_path,
            builder_name,
            scale,
            position,
        )

    @staticmethod
    def _heightmap_to_mesh(
        heightmap: np.ndarray,
        scale: float = 1.0,
        grid_size: float = 1.0,
        max_slope: float = np.deg2rad(89.0),  # 45 degrees by default
        smoothing: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a heightmap to a 3D mesh using NumPy operations.

        Args:
            heightmap: 2D numpy array of height values
            scale: Scale factor for heights
            grid_size: Base distance between grid points
            max_slope: Maximum allowed slope angle in radians
            smoothing: Whether to apply smoothing to the mesh

        Returns:
            vertices: 2D numpy array of vertex positions
            indices: 2D numpy array of triangle indices
        """
        rows, cols = heightmap.shape

        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(
            np.arange(cols) * grid_size, np.arange(rows) * grid_size
        )

        # Scale heights for Z coordinate
        z_coords = heightmap * scale

        # Handle maximum slope constraints
        if max_slope < np.pi / 2:
            # Calculate gradients
            dz_dx = np.zeros_like(heightmap)
            dz_dy = np.zeros_like(heightmap)

            dz_dx[:, 1:] = np.diff(z_coords, axis=1)
            dz_dy[1:, :] = np.diff(z_coords, axis=0)

            # Maximum allowed height difference based on slope
            max_diff = grid_size * np.tan(max_slope)

            # Clamp gradients
            dz_dx = np.clip(dz_dx, -max_diff, max_diff)
            dz_dy = np.clip(dz_dy, -max_diff, max_diff)

            # Reconstruct heights from clamped gradients
            z_coords = np.zeros_like(heightmap)
            z_coords[0, 0] = heightmap[0, 0] * scale

            # Integrate along x
            z_coords[0, 1:] = z_coords[0, 0] + np.cumsum(dz_dx[0, :-1])

            # Integrate along y
            for i in range(1, rows):
                z_coords[i, :] = z_coords[i - 1, :] + dz_dy[i - 1, :]

        # Optional smoothing
        if smoothing:
            import cv2

            kernel = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 8.0
            z_coords = cv2.filter2D(z_coords, -1, kernel)

        # Create vertices array with Z as height
        vertices = np.stack([x_coords, y_coords, z_coords], axis=-1).reshape(-1, 3)

        # Generate indices for triangles
        i = np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)
        i = i.reshape(-1)

        # For each grid cell, create two triangles
        triangles = np.array(
            [
                i,
                i + cols,
                i + 1,  # First triangle
                i + 1,
                i + cols,
                i + cols + 1,  # Second triangle
            ]
        )

        indices = triangles.T.reshape(-1, 3)

        return vertices.astype(np.float32), indices.astype(np.int32)

    @staticmethod
    def _add_mesh_to_world(
        vertices: np.ndarray,
        triangles: np.ndarray,
        base_path: str,
        builder_name: str,
        size: float,
        position: List[float],
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
            position[0] - size / 2,
            position[1] - size / 2,
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
            scale=[1.05, 1.05, 1.05],
            attributes={
                "points": vertices,
                "faceVertexIndices": triangles.flatten(),
                "faceVertexCounts": np.asarray([3] * num_faces),
                "subdivisionScheme": "bilinear",
            },
        )

        # ensure that we have all the necessary APIs
        collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
        collision_api.CreateCollisionEnabledAttr(True)

        return prim_path
