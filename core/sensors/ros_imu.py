from rclpy.qos import QoSProfile

from newton_sim_ros.msg import SimulationImuMsg
from newton_ros.msg import ImuMsg
from .imu import VecIMU as BaseVecIMU
from ..ros import BaseSimRealNode


class ROSVecIMU(BaseVecIMU, BaseSimRealNode):
    def __init__(
        self,
        vec_imu: BaseVecIMU,
        node_name: str = "imu_node",
        namespace: str = "sim",
        pub_sim_topic: str = "/sim/imu",
        pub_real_topic: str = "/imu",
        pub_period: int = 5,
        pub_qos_profile: QoSProfile = QoSProfile(depth=0),
    ):
        BaseVecIMU.__init__(
            self,
            vec_imu._universe,
            vec_imu.local_position,
            vec_imu.local_orientation,
            vec_imu._noise_function,
        )
        BaseSimRealNode.__init__(
            self,
            vec_imu._universe,
            node_name,
            SimulationImuMsg,
            pub_sim_topic,
            ImuMsg,
            pub_real_topic,
            namespace,
            pub_period,
            pub_qos_profile,
        )

    def construct(
        self,
        path_expr: str,
    ) -> None:
        BaseVecIMU.construct(self, path_expr)
        BaseSimRealNode.construct(self)

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

        self._indexed_publishers[self._pub_sim_topic].publish(msg)

        if self._pub_real_topic is not None:
            real_msg = ImuMsg()

            real_msg.header.stamp = msg.header.stamp
            real_msg.header.frame_id = msg.header.frame_id

            real_msg.rotation = msg.rotation
            real_msg.linear_velocity = msg.linear_velocity
            real_msg.angular_velocity = msg.angular_velocity
            real_msg.linear_acceleration = msg.linear_acceleration
            real_msg.angular_acceleration = msg.angular_acceleration

            self._indexed_publishers[self._pub_real_topic].publish(real_msg)
