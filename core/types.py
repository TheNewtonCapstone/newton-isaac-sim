import argparse
from typing import Dict, Any, Callable, List, Tuple, TypedDict, Optional

import torch


# Sensors
class IMUData(TypedDict):
    positions: torch.Tensor
    rotations: torch.Tensor
    linear_velocities: torch.Tensor
    linear_accelerations: torch.Tensor
    angular_accelerations: torch.Tensor
    angular_velocities: torch.Tensor
    projected_gravities: torch.Tensor


class ContactData(TypedDict):
    in_contacts: torch.Tensor
    contact_forces: torch.Tensor


# Motors

VecJointsPositions = torch.Tensor
VecJointsVelocities = torch.Tensor
VecJointsEfforts = torch.Tensor

VecJointPositionLimits = torch.Tensor
VecJointVelocityLimits = torch.Tensor
VecJointEffortLimits = torch.Tensor
VecJointGearRatios = torch.Tensor

JointPositionLimit = List[float]
JointVelocityLimit = float
JointEffortLimit = float

ArtJointsPositionLimits = Dict[str, List[float]]
ArtJointsVelocityLimits = Dict[str, float]
ArtJointsEffortLimits = Dict[str, float]
ArtJointsGearRatios = Dict[str, float]

JointPosition = float
JointVelocity = float
JointEffort = float
JointSaturation = float

# RL
Actions = torch.Tensor
Rewards = torch.Tensor
Dones = torch.Tensor
Progress = torch.Tensor

Infos = List[Dict[str, Any]]

EnvObservations = Dict[str, torch.Tensor]
Observations = torch.Tensor

# Math
NoiseFunction = Callable[[torch.Tensor], torch.Tensor]
# TODO: Create type for many types of numbers combined
#   I.e. float | torch.Tensor | np.ndarray

# Meta
Config = Dict[str, Any]
ConfigCollection = Dict[str, Config]
Indices = torch.Tensor

Matter = Tuple[
    argparse.Namespace,
    Config,
    Config,
    Config,
    Config,
    Config,
    ConfigCollection,
    str,
    str,
    ConfigCollection,
    Optional[str],
    str,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    int,
    float,
]
