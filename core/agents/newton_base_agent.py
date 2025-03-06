from abc import abstractmethod
from typing import List

import genesis as gs
from . import BaseAgent
from ..archiver import Archiver
from ..controllers import VecJointsController
from ..logger import Logger
from ..sensors import VecContact, VecIMU
from ..types import Actions, EnvObservations
from ..universe import Universe


class NewtonBaseAgent(BaseAgent):
    def __init__(
        self,
        universe: Universe,
        num_agents: int,
        imu: VecIMU,
        joints_controller: VecJointsController,
        contact_sensor: VecContact,
    ) -> None:
        super().__init__(
            universe=universe,
            num_agents=num_agents,
        )

        self.imu: VecIMU = imu
        self.joints_controller: VecJointsController = joints_controller
        self.contact_sensor: VecContact = contact_sensor

        self._create()

    @abstractmethod
    def step(self, actions: Actions) -> None:
        super().step(actions)

        self.joints_controller.step(actions)

        if self._universe.ros_enabled:
            from core.ros import BaseNode

            # we sync the step of the ROS nodes with the agent's step, therefore the physics step

            if isinstance(self.imu, BaseNode):
                self.imu.step()

            if isinstance(self.contact_sensor, BaseNode):
                self.contact_sensor.step()

    @abstractmethod
    def get_observations(self) -> EnvObservations:
        imu_data_tensor = self.imu.get_data()
        contact_data_tensor = self.contact_sensor.get_data()

        obs = {}
        first_obs = {}

        agent_count_median = self.num_agents // 2

        for key, value in imu_data_tensor.items():
            obs[key] = value
            first_obs[key] = value[agent_count_median, :]

        for key, value in contact_data_tensor.items():
            obs[key] = value
            first_obs[key] = value[agent_count_median, :]

        # we put only the first agent's observations
        Archiver.put("agent_obs", first_obs)

        return obs

    def _create(self) -> None:
        urdf_path = "assets/newton/newton.urdf"
        self._robot = self._universe.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
            )
        )

        self.imu.build(self._robot)
        self.joints_controller.build(self._robot)
        self.contact_sensor.build(self._robot)
