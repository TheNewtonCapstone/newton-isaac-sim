from rclpy.qos import QoSProfile

from carb.input import GamepadEvent, KeyboardEvent
from newton_ros.msg import CommandMsg
from newton_sim_ros.msg import SimulationCommandMsg
from .command_controller import CommandController as BaseCommandController
from ..ros import BaseSimRealNode


class ROSCommandController(BaseCommandController, BaseSimRealNode):
    def __init__(
        self,
        command_controller: BaseCommandController,
        node_name: str = "command_controller_node",
        namespace: str = "sim",
        pub_sim_topic: str = "/sim/command",
        pub_real_topic: str = "/command",
        pub_period: int = 5,
        pub_qos_profile: QoSProfile = QoSProfile(depth=0),
    ):
        BaseCommandController.__init__(
            self,
            command_controller._universe,
        )
        BaseSimRealNode.__init__(
            self,
            command_controller._universe,
            node_name,
            SimulationCommandMsg,
            pub_sim_topic,
            CommandMsg,
            pub_real_topic,
            namespace,
            pub_period,
            pub_qos_profile,
        )

    def post_build(
        self,
    ) -> None:
        BaseCommandController.post_build(self)
        BaseSimRealNode.post_build(self)

    def step(self) -> None:
        # No step function in BaseCommandController
        BaseSimRealNode.step(self)

    def publish(self) -> None:
        last_action = self.last_action
        current_triggers = self.current_triggers

        msg = SimulationCommandMsg()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"command_frame_{self._universe.current_time_step_index}"

        msg.target_velocity.x = last_action[0].item()
        msg.target_velocity.y = last_action[1].item()

        msg.triggers = current_triggers

        self._indexed_publishers[self._pub_sim_topic].publish(msg)

        if self._pub_real_topic is not None:
            real_msg = CommandMsg()

            real_msg.header.stamp = msg.header.stamp
            real_msg.header.frame_id = msg.header.frame_id

            real_msg.target_velocity = msg.target_velocity

            self._indexed_publishers[self._pub_real_topic].publish(real_msg)

    def _process_actions(self):
        processed_actions = BaseCommandController._process_actions(self)

        self.publish()

        return processed_actions
