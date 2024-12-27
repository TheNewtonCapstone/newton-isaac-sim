from newton_sim_ros.msg import SimulationContactsMsg
from .contact import VecContact as BaseVecContact
from ..ros import BaseNode
from ..universe import Universe


class ROSVecContact(BaseVecContact, BaseNode):
    def __init__(
        self,
        universe: Universe,
        num_contact_sensors_per_agent: int = 1,
        ros_node_name: str = "contact_node",
        namespace: str = "sim",
        publishing_topic: str = "/contact",
        publishing_frequency: float = 5.0,
    ):
        BaseVecContact.__init__(
            self,
            universe,
            num_contact_sensors_per_agent,
        )
        BaseNode.__init__(
            self,
            universe,
            SimulationContactsMsg,
            ros_node_name,
            publishing_topic,
            namespace,
            publishing_frequency,
        )

    def construct(
        self,
        path_expr: str,
    ) -> None:
        BaseVecContact.construct(self, path_expr)
        BaseNode.construct(self)

    def publish(self) -> None:
        data = BaseVecContact.get_data(self)
        msg = SimulationContactsMsg()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"contact_frame_{self._universe.current_time_step_index}"

        for i in range(self.num_contact_sensors_per_agent):
            msg.contacts.append(data["in_contacts"][0, i].item())
            msg.forces.append(data["contact_forces"][0, i].item())

        self._publisher.publish(msg)
