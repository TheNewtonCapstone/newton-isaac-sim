from abc import abstractmethod

import torch
from core.agents import BaseAgent
from core.controllers import VecJointsController
from core.sensors import VecIMU
from core.types import Actions, Observations
from core.universe import Universe


class NewtonBaseAgent(BaseAgent):
    def __init__(
        self,
        num_agents: int,
        imu: VecIMU,
        joints_controller: VecJointsController,
    ) -> None:
        super().__init__(num_agents)

        self.base_path_expr: str = ""

        self.imu: VecIMU = imu
        self.joints_controller: VecJointsController = joints_controller

    @abstractmethod
    def construct(self, universe: Universe) -> None:
        super().construct(universe)

    @abstractmethod
    def step(self, actions: Actions) -> None:
        super().step(actions)

        new_joint_positions = torch.from_numpy(actions).to(
            self._universe.physics_device
        )
        self.joints_controller.step(new_joint_positions)

    @abstractmethod
    def get_observations(self) -> Observations:
        imu_data_tensor = self.imu.get_data()
        imu_data_numpy = {}

        for key, value in imu_data_tensor.items():
            imu_data_numpy[key] = value.cpu().numpy()

        return imu_data_numpy
