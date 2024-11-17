from typing import Dict, Any, Callable

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

Infos = Dict[str, np.ndarray]

Observations = Dict[str, np.ndarray]

# Meta
Settings = Dict[str, Any]

# Math
NoiseFunction = Callable[[torch.Tensor], torch.Tensor]
