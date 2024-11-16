from typing import Dict, Any, Callable

import torch

# Sensory
IMUData = Dict[str, torch.FloatTensor]

# Motors
JointsPositions = torch.FloatTensor

# RL
Actions = torch.FloatTensor
Rewards = torch.FloatTensor
Dones = torch.BoolTensor
Progress = torch.FloatTensor

Infos = Dict[str, torch.Tensor]

Observations = Dict[str, torch.Tensor]

# Meta
Settings = Dict[str, Any]

# Math
NoiseFunction = Callable[[torch.FloatTensor], torch.FloatTensor]
