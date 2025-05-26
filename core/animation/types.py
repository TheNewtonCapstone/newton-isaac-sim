from typing import Dict, List

import torch
from dataclasses import dataclass, field

@dataclass
class BoneData:
    name: str
    position: torch.Tensor
    orientation: torch.Tensor

    relative_angle: float
    relative_angle_velocity: float


ArmatureData = Dict[str, BoneData]


@dataclass
class Keyframe:
    frame: float
    data: ArmatureData


@dataclass
class AnimationClip:
    name: str
    framerate: int
    duration: int
    duration_in_seconds: float
    keyframes: List[Keyframe]

