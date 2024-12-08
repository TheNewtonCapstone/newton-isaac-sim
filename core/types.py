from typing import Dict, Any, Callable, List, Optional

import numpy as np
import torch

# Sensory
IMUData = Dict[str, torch.Tensor]

# Motors
JointsPositions = torch.Tensor

# RL
Actions = np.ndarray
Rewards = np.ndarray
Dones = np.ndarray
Progress = np.ndarray

Infos = List[Dict[str, Any]]

Observations = Dict[str, np.ndarray]

# Meta
Settings = Dict[str, Any]
# TODO: removed Optional, none of the global types should be Optional (can easily lead to confusion)
Indices = Optional[np.ndarray]

# Math
NoiseFunction = Callable[[torch.Tensor], torch.Tensor]
