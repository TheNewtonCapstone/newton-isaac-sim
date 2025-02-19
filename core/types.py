import argparse
from typing import Dict, Any, Callable, List, Tuple, TypedDict, Optional, Literal

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
VecJointFixed = torch.Tensor

JointPositionLimit = List[float]
JointVelocityLimit = float
JointEffortLimit = float

ArtJointsPositionLimits = Dict[str, List[float]]
ArtJointsVelocityLimits = Dict[str, float]
ArtJointsEffortLimits = Dict[str, float]
ArtJointsGearRatios = Dict[str, float]
ArtJointsFixed = Dict[str, bool]

JointPosition = float
JointVelocity = float
JointEffort = float
JointSaturation = float

# RL
Observations = torch.Tensor
Actions = torch.Tensor
Rewards = torch.Tensor
Dones = torch.Tensor
Terminated = torch.Tensor
Truncated = torch.Tensor
EpisodeLength = torch.Tensor

EnvObservations = Dict[str, torch.Tensor]


class Extras(TypedDict):
    episode: Dict[str, torch.Tensor]
    time_outs: torch.Tensor


StepReturn = Tuple[Observations, Rewards, Terminated, Truncated, Extras]
ResetReturn = Tuple[Observations, Extras]
TaskObservations = Tuple[Observations, Extras]

ObservationScalers = Dict[str, float]
RewardScalers = Dict[str, float]
ActionScaler = float
CommandScalers = Dict[str, float]

# Math
NoiseFunction = Callable[[torch.Tensor], torch.Tensor]

# Meta
Config = Dict[str, Any]
ConfigCollection = Dict[str, Config]
Indices = torch.Tensor
Mode = Literal["training", "playing", "animating", "physics-only", "exporting"]


class Matter(TypedDict):
    cli_args: argparse.Namespace
    robot_config: Config
    task_configs: ConfigCollection
    current_task_config: Config
    current_task_name: Optional[str]
    world_config: Config
    randomization_config: Config
    network_configs: ConfigCollection
    current_network_config: Config
    current_network_name: Optional[str]
    ros_config: Config
    db_config: Config
    logger_config: Config
    log_file_path: str
    terrain_config: Config
    secrets: Config
    animation_clips_config: ConfigCollection
    current_animation_clip_config: Config
    current_animation: Optional[str]
    runs_dir: str
    runs_library: Config
    current_run_name: Optional[str]
    current_checkpoint_path: Optional[str]
    mode: Mode
    mode_name: str
    training: bool
    playing: bool
    animating: bool
    physics_only: bool
    exporting: bool
    is_rl: bool
    interactive: bool
    headless: bool
    enable_ros: bool
    enable_db: bool
    num_envs: int
    control_step_dt: float
    inverse_control_frequency: int


# System


class CallerInfo(TypedDict):
    filename: str
    lineno: int
    funcname: str
    modulename: str
