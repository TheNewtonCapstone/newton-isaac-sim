def create_physics_material(
    material_prim_path: str,
    dynamic_friction: float = 0.5,
    static_friction: float = 0.5,
    restitution: float = 0.8
):
    from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
    from core.utils.usd import get_or_apply_api
    from pxr import UsdPhysics

    assert is_prim_path_valid(material_prim_path), f"Invalid prim path: {material_prim_path}"

    physics_material_api = get_or_apply_api(get_prim_at_path(material_prim_path), UsdPhysics.MaterialAPI)

    physics_material_api.CreateDynamicFrictionAttr(dynamic_friction)
    physics_material_api.CreateStaticFrictionAttr(static_friction)
    physics_material_api.CreateRestitutionAttr(restitution)


def set_physics_properties(
    target_prim_path: str,
    dynamic_friction: float = 0.5,
    static_friction: float = 0.5,
    restitution: float = 0.8
):
    from omni.isaac.core.utils.prims import get_prim_at_path, define_prim, is_prim_path_valid

    target_prim = get_prim_at_path(target_prim_path)

    assert target_prim.IsValid(), f"Invalid target_prim: {target_prim_path}"

    from core.utils.usd import get_or_apply_api, get_or_define_material
    from pxr import UsdPhysics, UsdShade

    material_prim_path = f"{target_prim_path}/physics_material"

    if not is_prim_path_valid(material_prim_path):
        material_prim = define_prim(material_prim_path, "Material")
    else:
        material_prim = get_prim_at_path(material_prim_path)

    # if the material has already been bound to the target prim, just update the physics properties
    if target_prim.HasAPI(UsdShade.MaterialBindingAPI):
        physics_material_api = get_or_apply_api(material_prim, UsdPhysics.MaterialAPI)
        physics_material_api.CreateDynamicFrictionAttr(dynamic_friction)
        physics_material_api.CreateStaticFrictionAttr(static_friction)
        physics_material_api.CreateRestitutionAttr(restitution)

        return

    # otherwise, bind the material (not the material prim) to the target prim
    material = get_or_define_material(material_prim_path)

    material_binding_api = get_or_apply_api(target_prim, UsdShade.MaterialBindingAPI)
    material_binding_api.Bind(
        material, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics"
    )

    physics_material_api = get_or_apply_api(material_prim, UsdPhysics.MaterialAPI)
    physics_material_api.CreateDynamicFrictionAttr(dynamic_friction)
    physics_material_api.CreateStaticFrictionAttr(static_friction)
    physics_material_api.CreateRestitutionAttr(restitution)

    return


def raycast(start: list[float], direction: list[float], max_distance: float = 500) -> tuple[list[float], list[float], float]:
    """
    Raycast from a start position in a direction with a maximum distance.

    Args:
        start:
        direction:
        max_distance:

    Returns:
        Tuple with the position of the hit, normal of the hit surface, and distance to the hit. Distance is -1 if no hit.
    """

    from omni.physx import get_physx_scene_query_interface
    from pxr import Gf

    origin = Gf.Vec3f(start)
    ray_dir = Gf.Vec3f(direction)
    max_dist = max_distance

    hit_info = get_physx_scene_query_interface().raycast_closest(origin, ray_dir, max_dist, True)

    if hit_info is None or not hit_info["hit"]:
        return [], [], -1

    hit_position = list(hit_info["position"])
    hit_normal = list(hit_info["normal"])
    hit_distance = hit_info["distance"]

    return hit_position, hit_normal, hit_distance
