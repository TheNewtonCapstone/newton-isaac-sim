from core.types import (
    ArtJointsPositionLimits,
    ArtJointsVelocityLimits,
    ArtJointsEffortLimits,
    VecJointEffortLimits,
    VecJointVelocityLimits,
    VecJointPositionLimits,
    ArtJointsGearRatios,
    VecJointGearRatios,
    ArtJointsFixed,
    VecJointFixed,
)


def dict_to_vec_limits(
    joint_limits: (
        ArtJointsPositionLimits
        | ArtJointsVelocityLimits
        | ArtJointsEffortLimits
        | ArtJointsGearRatios
        | ArtJointsFixed
    ),
    device: str,
) -> (
    VecJointPositionLimits
    | VecJointVelocityLimits
    | VecJointEffortLimits
    | VecJointGearRatios
    | VecJointFixed
):
    joint_names = list(joint_limits.keys())

    import torch

    # Check if the joint limits are in the form of a list of lists or a list of floats (position requires 2 values)
    if isinstance(list(joint_limits.values())[0], list):
        vec_joint_limits = torch.zeros((len(joint_names), 2))
    else:
        vec_joint_limits = torch.zeros((len(joint_names), 1))

    # Ensures that the joint constraints are in the correct order
    for i, joint_name in enumerate(joint_names):
        limits = joint_limits[joint_name]

        vec_joint_limits[i, :] = torch.tensor(limits)

    return vec_joint_limits.to(device=device)
