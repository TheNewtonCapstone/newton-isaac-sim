from abc import abstractmethod
from typing import Optional

import numpy as np

import torch
from core.agents import BaseAgent
from core.sensors import VecIMU
from core.controllers import VecJointsController
from core.types import Actions, Observations
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView


class NewtonBaseAgent(BaseAgent):
    def __init__(self, num_agents: int):
        super().__init__(num_agents)

        self.base_path_expr: str = ""

        self.imu: Optional[VecIMU] = None
        self.joints_controller: Optional[VecJointsController] = None

    @abstractmethod
    def construct(self, world: World) -> None:
        super().construct(world)

    @abstractmethod
    def step(self, actions: Actions) -> None:
        super().step(actions)

        new_joint_positions = torch.from_numpy(actions).to(self.world.device)
        self.joints_controller.step(new_joint_positions)

    @abstractmethod
    def get_observations(self) -> Observations:
        imu_data_tensor = self.imu.get_data(recalculate=True)
        imu_data_numpy = {}

        for key, value in imu_data_tensor.items():
            imu_data_numpy[key] = value.cpu().numpy()

        return imu_data_numpy

    def _construct_imu(self, path_expr: str):
        from core.utils.math import IDENTITY_QUAT

        self.imu = VecIMU(
            path_expr=path_expr,
            world=self.world,
            local_position=torch.zeros((self.num_agents, 3)),
            local_rotation=IDENTITY_QUAT.repeat(self.num_agents, 1),
            noise_function=lambda x: x,
        )
        self.imu.construct()

    def _construct_joints_controller(self, path_expr: str):
        from gymnasium.spaces import Box

        self.joints_controller = VecJointsController(
            path_expr=path_expr,
            world=self.world,
            noise_function=lambda x: x,
            joint_constraints=Box(
                low=np.array([-15, -90, -180] * 4),
                high=np.array([15, 90, 180] * 4),
            ),
        )
        self.joints_controller.construct()
