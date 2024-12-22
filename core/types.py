import argparse
from typing import Dict, Any, Callable, List, Optional, Tuple, TypedDict

import numpy as np
import torch

# Sensors
IMUData = Dict[str, torch.Tensor]


class ContactData(TypedDict):
    in_contacts: torch.Tensor
    contact_forces: torch.Tensor


# Motors
JointsPositions = torch.Tensor
JointsConstraints = Dict[str, Tuple[float, float]]

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
Indices = torch.Tensor

Matter = Tuple[argparse.Namespace, Settings, Settings, Settings, Settings, Dict[
    str, Settings], str, bool, bool, bool, bool, bool, bool, bool, bool, int, float]
