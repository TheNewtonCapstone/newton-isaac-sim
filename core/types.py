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
JointsPositions = torch.Tensor
JointsPositionLimits = Dict[str, List[float]]
JointsVelocityLimits = Dict[str, float]

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
Settings = Dict[str, Any]
SettingsCollection = Dict[str, Settings]
Indices = torch.Tensor

Matter = Tuple[
    argparse.Namespace,
    Settings,
    Settings,
    Settings,
    Settings,
    SettingsCollection,
    str,
    str,
    SettingsCollection,
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
