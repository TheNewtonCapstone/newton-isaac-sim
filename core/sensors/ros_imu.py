from torch import Tensor

from newton_sim_ros.msg import SimulationImuMsg
from .imu import VecIMU as BaseVecIMU
from ..ros import BaseNode
from ..types import NoiseFunction
from ..universe import Universe


class ROSVecIMU(BaseVecIMU, BaseNode):
    def __init__(
        self,
        universe: Universe,
        local_position: Tensor,
        local_orientation: Tensor,
        noise_function: NoiseFunction,
        ros_node_name: str = "imu_node",
        namespace: str = "sim",
        publishing_topic: str = "/imu",
        publishing_frequency: float = 5.0,
    ):
        BaseVecIMU.__init__(
            self,
            universe,
            local_position,
            local_orientation,
            noise_function,
        )
        BaseNode.__init__(
            self,
            universe,
            SimulationImuMsg,
            ros_node_name,
            publishing_topic,
            namespace,
            publishing_frequency,
        )

    def construct(
        self,
        path_expr: str,
    ) -> None:
        BaseVecIMU.construct(self, path_expr)
        BaseNode.construct(self)

    def publish(self) -> None:
        data = BaseVecIMU.get_data(self)
        msg = SimulationImuMsg()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"imu_frame_{self._universe.current_time_step_index}"
        msg.position.x = data["positions"][0, 0].item()
        msg.position.y = data["positions"][0, 1].item()
        msg.position.z = data["positions"][0, 2].item()
        msg.rotation.x = data["rotations"][0, 0].item()
        msg.rotation.y = data["rotations"][0, 1].item()
        msg.rotation.z = data["rotations"][0, 2].item()
        msg.linear_velocity.x = data["linear_velocities"][0, 0].item()
        msg.linear_velocity.y = data["linear_velocities"][0, 1].item()
        msg.linear_velocity.z = data["linear_velocities"][0, 2].item()
        msg.angular_velocity.x = data["angular_velocities"][0, 0].item()
        msg.angular_velocity.y = data["angular_velocities"][0, 1].item()
        msg.angular_velocity.z = data["angular_velocities"][0, 2].item()
        msg.linear_acceleration.x = data["linear_accelerations"][0, 0].item()
        msg.linear_acceleration.y = data["linear_accelerations"][0, 1].item()
        msg.linear_acceleration.z = data["linear_accelerations"][0, 2].item()
        msg.angular_acceleration.x = data["angular_accelerations"][0, 0].item()
        msg.angular_acceleration.y = data["angular_accelerations"][0, 1].item()
        msg.angular_acceleration.z = data["angular_accelerations"][0, 2].item()
        msg.projected_gravity.x = data["projected_gravities"][0, 0].item()
        msg.projected_gravity.y = data["projected_gravities"][0, 1].item()
        msg.projected_gravity.z = data["projected_gravities"][0, 2].item()

        self._publisher.publish(msg)
