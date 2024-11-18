from abc import abstractmethod
from typing import List

from core.globals import LIGHTS_PATH
from torch import Tensor

import numpy as np
import torch
from core.agents import NewtonBaseAgent
from core.envs import BaseEnv
from core.terrain import TerrainBuilder
from core.types import Observations, Settings


class NewtonBaseEnv(BaseEnv):
    def __init__(
        self,
        agent: NewtonBaseAgent,
        num_envs: int,
        terrain_builders: List[TerrainBuilder],
        world_settings: Settings,
        randomizer_settings: Settings,
    ):
        super().__init__(
            agent,
            num_envs,
            terrain_builders,
            world_settings,
            randomizer_settings,
        )

        self.agent: NewtonBaseAgent = agent

        self.reset_newton_positions: Tensor = torch.zeros((self.num_envs, 3))
        self.reset_newton_rotations: Tensor = torch.tile(
            torch.tensor([0.0, 0.0, 0.0, 1.0]), (self.num_envs, 1)
        )

    @abstractmethod
    def construct(self) -> None:
        super().construct()

        # add light to the scene
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.rotations import euler_angles_to_quat

        create_prim(
            prim_path=f"{LIGHTS_PATH}/WorldLight",
            prim_type="DistantLight",
            orientation=euler_angles_to_quat(np.array([40, 0, 40]), True),
            attributes={
                "inputs:intensity": 3e3,
                "inputs:color": (0.93, 0.84, 0.62),
            },
        )

        create_prim(
            prim_path=f"{LIGHTS_PATH}/WorldAntiLight",
            prim_type="DistantLight",
            orientation=euler_angles_to_quat(np.array([40, 0, 140]), True),
            attributes={
                "inputs:intensity": 3e3,
                "inputs:color": (0.63, 0.84, 0.92),
            },
        )

    @abstractmethod
    def step(self, actions: torch.Tensor, render: bool) -> Observations:
        super().step(actions, render)  # advances the simulation by one step

        return self.get_observations()

    @abstractmethod
    def reset(self) -> Observations:
        super().reset()

        return self.get_observations()

    @abstractmethod
    def get_observations(self) -> Observations:
        return super().get_observations()
