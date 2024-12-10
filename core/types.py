import argparse
from typing import Dict, Any, Callable, List, Optional, Tuple

import numpy as np
import torch

# Sensory
IMUData = Dict[str, torch.Tensor]

# Motors
JointsPositions = torch.Tensor
JointsConstraints = Dict[str, Tuple[float, float]]

# RL
Actions = np.ndarray
Rewards = np.ndarray
Dones = np.ndarray
Progress = np.ndarray

Infos = List[Dict[str, Any]]

Observations = Dict[str, np.ndarray]

# Math
NoiseFunction = Callable[[torch.Tensor], torch.Tensor]

# Meta
Settings = Dict[str, Any]
# TODO: removed Optional, none of the global types should be Optional (can easily lead to confusion)
Indices = Optional[np.ndarray]

Matter = Tuple[argparse.Namespace, Settings, Settings, Settings, Dict[str, Settings], str, bool, bool, bool, bool, bool, bool, bool, bool, int, float]
