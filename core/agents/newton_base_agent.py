from abc import abstractmethod
from typing import Optional

import torch
from core.agents import BaseAgent
from core.newton.imu import VecIMU
from core.newton.joints_controller import VecJointsController
from core.types import IMUData


class NewtonBaseAgent(BaseAgent):
    def __init__(self, num_agents: int, world):
        super().__init__(num_agents, world)

        self.imu: Optional[VecIMU] = None
        self.joints_controller: Optional[VecJointsController] = None

        self._is_constructed: bool = False

    @abstractmethod
    def construct(self, root_path: str) -> str:
        pass

    def _construct_imu(self, path_expr: str):
        self.imu = VecIMU(
            path_expr=path_expr,
            local_position=torch.zeros((self.num_agents, 3)),
            local_rotation=torch.tile(
                torch.tensor([0.0, 0.0, 0.0, 1.0]), (self.num_agents, 1)
            ),
            noise_function=lambda x: x,
        )
        self.imu.construct()

    def _construct_joints_controller(self, path_expr: str):
        self.joints_controller = VecJointsController(
            path_expr=path_expr,
            noise_function=lambda x: x,
        )
        self.joints_controller.construct()
