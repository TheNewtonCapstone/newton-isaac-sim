from rclpy.qos import QoSProfile
from torch import Tensor

from newton_ros.msg import JointsMsg
from newton_sim_ros.msg import SimulationJointsMsg
from .joints_controller import VecJointsController as BaseVecJointsController
from core.ros import BaseSimRealNode


class ROSVecJointsController(BaseVecJointsController, BaseSimRealNode):
    def __init__(
        self,
        vec_joints_controller: BaseVecJointsController,
        node_name: str = "joints_controller_node",
        namespace: str = "sim",
        pub_sim_topic: str = "/sim/joints",
        pub_real_topic: str = "/joints",
        pub_period: int = 5,
        pub_qos_profile: QoSProfile = QoSProfile(depth=0),
    ):
        BaseVecJointsController.__init__(
            self,
            vec_joints_controller._universe,
            vec_joints_controller._noise_function,
            vec_joints_controller._joint_position_limits,
            vec_joints_controller._joint_velocity_limits,
            vec_joints_controller._joint_effort_limits,
            vec_joints_controller._gear_ratios,
            vec_joints_controller._actuators,
            vec_joints_controller._fixed_joints,
        )
        BaseSimRealNode.__init__(
            self,
            vec_joints_controller._universe,
            node_name,
            SimulationJointsMsg,
            pub_sim_topic,
            JointsMsg,
            pub_real_topic,
            namespace,
            pub_period,
            pub_qos_profile,
        )

    def construct(
        self,
        path_expr: str,
    ) -> None:
        BaseVecJointsController.construct(self, path_expr)
        BaseSimRealNode.construct(self)

    def step(self, joint_actions: Tensor) -> None:
        BaseVecJointsController.step(self, joint_actions)
        BaseSimRealNode.step(self)

    def publish(self) -> None:
        target_positions = BaseVecJointsController.get_target_joint_positions_deg(self)
        positions = BaseVecJointsController.get_joint_positions_deg(self)
        velocities = BaseVecJointsController.get_joint_velocities_deg(self)
        applied_torques = BaseVecJointsController.get_applied_joint_efforts(self)

        msg = SimulationJointsMsg()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"joints_frame_{self._universe.current_time_step_index}"

        msg.target_positions = target_positions[0].tolist()
        msg.positions = positions[0].tolist()
        msg.velocities = velocities[0].tolist()
        msg.applied_torques = applied_torques[0].tolist()

        self._indexed_publishers[self._pub_sim_topic].publish(msg)

        if self._pub_real_topic is not None:
            real_msg = JointsMsg()

            real_msg.header.stamp = msg.header.stamp
            real_msg.header.frame_id = msg.header.frame_id

            real_msg.target_positions = msg.target_positions
            real_msg.positions = msg.positions
            real_msg.velocities = msg.velocities
            real_msg.applied_torques = msg.applied_torques

            self._indexed_publishers[self._pub_real_topic].publish(real_msg)
