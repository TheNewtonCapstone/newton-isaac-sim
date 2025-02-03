import re
from typing import Any, List, Optional

from pxr.UsdShade import Material

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Usd, Gf, UsdGeom


def get_or_define_material(material_prim_path: str) -> Material:
    """Get or define a material at the specified prim path.

    Args:
        material_prim_path: The prim path of the material.

    Returns:
        A tuple containing the material prim and the material API.

    Raises:
        ValueError: If the material API cannot be applied to the prim.
    """

    from omni.isaac.core.utils.prims import is_prim_path_valid
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdShade

    stage = get_current_stage()

    # define material prim
    if not is_prim_path_valid(material_prim_path):
        return UsdShade.Material(stage.GetPrimAtPath(material_prim_path))
    else:
        return UsdShade.Material.Define(stage, material_prim_path)


def get_or_apply_api(prim: Usd.Prim, api_type: Any) -> Usd.APISchemaBase:
    """Get or apply the API to the prim.

    Args:
        prim: The prim to get or apply the API to.
        api_type: The API type to apply to the prim.

    Returns:
        The API applied to the prim.

    Raises:
        ValueError: If the API cannot be applied to the prim.
    """

    from omni.isaac.core.utils.stage import get_current_stage

    # get API if it already exists
    api = api_type.Get(get_current_stage(), prim.GetPath())
    if api:
        return api

    # apply API
    api = api_type.Apply(prim)
    if not api:
        raise ValueError(
            f"Failed to apply API '{api_type.__name__}' to prim '{prim.GetPath()}'."
        )

    return api


def find_matching_prims(prim_path_regex: str) -> list[Usd.Prim]:
    """Find all the matching prims in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.

    Returns:
        A list of prims that match input expression.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """

    from omni.isaac.core.utils.stage import get_current_stage

    # check prim path is global
    if not prim_path_regex.startswith("/"):
        raise ValueError(
            f"Prim path '{prim_path_regex}' is not global. It must start with '/'."
        )

    # get current stage
    stage = get_current_stage()

    # need to wrap the token patterns in '^' and '$' to prevent matching anywhere in the string
    tokens = prim_path_regex.split("/")[1:]
    tokens = [f"^{token}$" for token in tokens]

    # iterate over all prims in stage (breath-first search)
    all_prims = [stage.GetPseudoRoot()]
    output_prims = []
    for index, token in enumerate(tokens):
        token_compiled = re.compile(token)
        for prim in all_prims:
            for child in prim.GetAllChildren():
                if token_compiled.match(child.GetName()) is not None:
                    output_prims.append(child)
        if index < len(tokens) - 1:
            all_prims = output_prims
            output_prims = []

    return output_prims


def find_matching_prims_count(prim_path_regex: str) -> int:
    """Find the number of matching prims in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The number of prims that match input expression.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """

    return len(find_matching_prims(prim_path_regex))


def path_expr_to_paths(
    root_path_expr: str,
    child_path_expr: str,
) -> List[List[str]]:
    """Convert a path expression to a list of paths (2-deep).

    Args:
        root_path_expr: Path expression for the root prims.
        child_path_expr: Path expression for the child prims (within the root).

    Returns:
        A list of lists of paths, where each inner list contains the paths of the children of a root prim,
        in order of discovery.
    """
    root_prims_count = find_matching_prims_count(root_path_expr)
    children_prims_paths = find_matching_prim_paths(root_path_expr + child_path_expr)
    children_prims_count = len(children_prims_paths)
    children_per_root_count = children_prims_count // root_prims_count

    full_prim_paths = [[] for _ in range(root_prims_count)]

    for i in range(len(children_prims_paths)):
        full_prim_paths[i // children_per_root_count].append(children_prims_paths[i])

    return full_prim_paths


def get_parent_prim_from_path(prim_path: str) -> Usd.Prim:
    """Get the parent prim from the prim path.

    Args:
        prim_path: The prim path.

    Returns:
        The parent prim of the prim path.
    """

    from omni.isaac.core.utils.stage import get_current_stage

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)

    return prim.GetParent()


def get_parent_path_from_path(prim_path: str) -> str:
    """Get the parent path from the prim path.

    Args:
        prim_path: The prim path.

    Returns:
        The parent path of the prim path.
    """

    return get_parent_prim_from_path(prim_path).GetPath().pathString


def get_parent_expr_path_from_expr_path(prim_path: str) -> str:
    """Get the parent path from a prim's expression path.

    Args:
        prim_path: The prim's expression path; the prim does not need to exist.

    Returns:
        The parent's expression path of the prim path.
    """

    return "/".join(prim_path.split("/")[:-1])


def get_prim_bounds_range(
    prim_path: str,
    bbox_cache: Optional[UsdGeom.BBoxCache] = None,
) -> Gf.Range3d:
    """Get the extent of the prim.

    Args:
        prim: The prim to get the extent of.

    Returns:
        The extent of the prim.
    """

    import omni.timeline

    from pxr import UsdGeom

    timeline = omni.timeline.get_timeline_interface()
    stage = get_current_stage()
    current_timecode = timeline.get_current_time()

    if bbox_cache is None:
        bbox_cache = UsdGeom.BBoxCache(
            current_timecode,
            includedPurposes=[UsdGeom.Tokens.default_],
        )

    bbox_cache.Clear()
    bbox_cache.SetTime(current_timecode - stage.GetStartTimeCode())

    agent_prim = get_prim_at_path(prim_path)
    bounds = bbox_cache.ComputeWorldBound(agent_prim)

    return bounds.GetRange()
