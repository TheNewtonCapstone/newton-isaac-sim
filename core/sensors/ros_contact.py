from newton_ros.msg import ContactsMsg
from rclpy.qos import QoSProfile

from newton_sim_ros.msg import SimulationContactsMsg
from .contact import VecContact as BaseVecContact
from ..ros import BaseSimRealNode


class ROSVecContact(BaseVecContact, BaseSimRealNode):
    def __init__(
        self,
        vec_contact: BaseVecContact,
        node_name: str = "contact_node",
        namespace: str = "sim",
        pub_sim_topic: str = "/sim/contact",
        pub_real_topic: str = "/contact",
        pub_period: int = 5,
        pub_qos_profile: QoSProfile = QoSProfile(depth=0),
    ):
        BaseVecContact.__init__(
            self,
            vec_contact._universe,
            vec_contact._num_contact_sensors_per_agent,
        )
        BaseSimRealNode.__init__(
            self,
            vec_contact._universe,
            node_name,
            SimulationContactsMsg,
            pub_sim_topic,
            ContactsMsg,
            pub_real_topic,
            namespace,
            pub_period,
            pub_qos_profile,
        )

    def construct(
        self,
        path_expr: str,
    ) -> None:
        BaseVecContact.construct(self, path_expr)
        BaseSimRealNode.construct(self)

    def publish(self) -> None:
        data = BaseVecContact.get_data(self)
        sim_msg = SimulationContactsMsg()

        sim_msg.header.stamp = self.get_clock().now().to_msg()
        sim_msg.header.frame_id = (
            f"contact_frame_{self._universe.current_time_step_index}"
        )

        for i in range(self._num_contact_sensors_per_agent):
            sim_msg.contacts.append(data["in_contacts"][0, i].item())
            sim_msg.forces.append(data["contact_forces"][0, i].item())

        self._indexed_publishers[self._pub_sim_topic].publish(sim_msg)

        if self._pub_real_topic is not None:
            real_msg = ContactsMsg()

            real_msg.header.stamp = sim_msg.header.stamp
            real_msg.header.frame_id = sim_msg.header.frame_id

            real_msg.contacts = sim_msg.contacts

            self._indexed_publishers[self._pub_real_topic].publish(real_msg)
