import re
from typing import Any
from pxr import Usd
from pxr.UsdShade import Material


def get_or_define_material(material_prim_path: str) -> Material:
    """Get or define a material at the specified prim path.

    Args:
        material_prim_path: The prim path of the material.

    Returns:
        A tuple containing the material prim and the material API.

    Raises:
        ValueError: If the material API cannot be applied to the prim.
    """

    from omni.isaac.core.utils.prims import define_prim, is_prim_path_valid
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
    from pxr import Usd

    # get API if it already exists
    api = api_type.Get(get_current_stage(), prim.GetPath())
    if api:
        return api

    # apply API
    api = api_type.Apply(prim)
    if not api:
        raise ValueError(f"Failed to apply API '{api_type.__name__}' to prim '{prim.GetPath()}'.")

    return api


def find_matching_prims(prim_path_regex: str, stage: Any | None = None) -> list[Any]:
    """Find all the matching prims in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        A list of prims that match input expression.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """

    from omni.isaac.core.utils.stage import get_current_stage

    # check prim path is global
    if not prim_path_regex.startswith("/"):
        raise ValueError(f"Prim path '{prim_path_regex}' is not global. It must start with '/'.")

    # get current stage
    if stage is None:
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
