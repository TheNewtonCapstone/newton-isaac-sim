from typing import List, Optional, Tuple

import numpy as np
import torch as th


def add_heightmap_to_world(
    heightmap: np.ndarray | th.Tensor,
    scale: float,
    height: float,
    base_path: str,
    slope_threshold: float,
    position: List[float],
) -> str:
    vertices, triangles = heightmap_to_mesh(
        heightmap,
        scale,
        height,
        slope_threshold,
    )

    return add_mesh_to_world(
        vertices,
        triangles,
        base_path,
        position,
    )


def heightmap_to_mesh(
    height_field_raw: np.ndarray,
    horizontal_resolution: float,
    vertical_resolution: float,
    slope_threshold: Optional[float] = None,
) -> Tuple[np.ndarray[float], np.ndarray[int]]:
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw: Input heightfield array of shape (num_rows, num_cols).
        horizontal_resolution: Horizontal scale of the heightfield (in meters).
        vertical_resolution: vertical scale of the heightfield (in meters).
        slope_threshold: the slope threshold above which surfaces are made vertical. If None, no correction is applied.
    Returns:
        vertices: array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles: array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_resolution, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_resolution, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        slope_threshold *= horizontal_resolution / vertical_resolution

        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))

        move_x[: num_rows - 1, :] += (
            hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        )
        move_x[1:num_rows, :] -= (
            hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        )

        move_y[:, : num_cols - 1] += (
            hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        )
        move_y[:, 1:num_cols] -= (
            hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        )

        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1]
            > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols]
            > slope_threshold
        )

        xx += (move_x + move_corners * (move_x == 0)) * horizontal_resolution
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_resolution

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_resolution
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)

    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1

        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)

        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return vertices, triangles


def add_mesh_to_world(
    vertices: np.ndarray,
    triangles: np.ndarray,
    base_path: str,
    position: List[float],
) -> str:
    from core.utils.usd import find_matching_prims
    from omni.isaac.core.utils.prims import (
        define_prim,
        create_prim,
        is_prim_path_valid,
    )
    from pxr import UsdPhysics, PhysxSchema

    # generate an informative and unique name from the type of builder
    group_prim_path = base_path
    prim_path_expr = f"{group_prim_path}/terrain_.*"
    num_of_existing_terrains = len(find_matching_prims(prim_path_expr))
    prim_path = f"{group_prim_path}/terrain_{num_of_existing_terrains}"
    num_faces = triangles.shape[0]

    if not is_prim_path_valid(group_prim_path):
        define_prim(
            group_prim_path,
            prim_type="Scope",
        )

    # creates the terrain's root prim
    create_prim(
        prim_path,
        prim_type="Xform",
        position=position,
    )

    # creates the mesh prim, that actually collides
    mesh_prim = create_prim(
        prim_path + "/mesh",
        prim_type="Mesh",
        scale=[1.0, 1.0, 1.0],
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

    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    physx_collision_api.GetContactOffsetAttr().Set(0.02)
    physx_collision_api.GetRestOffsetAttr().Set(0.00)

    return prim_path
