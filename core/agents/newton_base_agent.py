from abc import abstractmethod

from core.agents import BaseAgent
from core.controllers import VecJointsController
from core.sensors import VecIMU, VecContact
from core.types import Actions, EnvObservations
from core.universe import Universe

# TODO: Add a command controller to the NewtonBaseAgent
#   It would allow for control of the agent's movement through a keyboard or a controller. Any Task would be able to
#   read the commands and use them as they see fit (i.e. to train).


class NewtonBaseAgent(BaseAgent):
    def __init__(
        self,
        num_agents: int,
        imu: VecIMU,
        joints_controller: VecJointsController,
        contact_sensor: VecContact,
    ) -> None:
        super().__init__(num_agents)

        self.base_path_expr: str = ""

        self.imu: VecIMU = imu
        self.joints_controller: VecJointsController = joints_controller
        self.contact_sensor: VecContact = contact_sensor

    @abstractmethod
    def construct(self, universe: Universe) -> None:
        super().construct(universe)

    @abstractmethod
    def step(self, actions: Actions) -> None:
        super().step(actions)

        self.joints_controller.step(actions)

    @abstractmethod
    def get_observations(self) -> EnvObservations:
        imu_data_tensor = self.imu.get_data()
        contact_data_tensor = self.contact_sensor.get_data()
        data_numpy = {}

        for key, value in imu_data_tensor.items():
            data_numpy[key] = value

        for key, value in contact_data_tensor.items():
            data_numpy[key] = value

        return data_numpy
