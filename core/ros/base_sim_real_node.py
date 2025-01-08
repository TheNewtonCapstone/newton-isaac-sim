from abc import abstractmethod
from typing import Type, Optional

from rclpy.qos import QoSProfile

from . import BaseNode
from ..universe import Universe


class BaseSimRealNode(BaseNode):
    def __init__(
        self,
        universe: Universe,
        node_name: str,
        pub_sim_msg_type: Type,
        pub_sim_topic: str,
        pub_real_msg_type: Optional[Type] = None,
        pub_real_topic: Optional[str] = None,
        namespace: str = "sim",
        pub_period: int = 5,
        pub_qos_profile: QoSProfile = QoSProfile(depth=0),
    ):
        publishing_topics = (
            [pub_sim_topic, pub_real_topic]
            if pub_real_topic is not None
            else [pub_sim_topic]
        )
        publishing_types = (
            [pub_sim_msg_type, pub_real_msg_type]
            if pub_real_topic is not None
            else [pub_sim_msg_type]
        )

        BaseNode.__init__(
            self,
            universe,
            node_name,
            publishing_types,
            publishing_topics,
            namespace,
            pub_period,
            pub_qos_profile,
        )

        self._pub_real_topic: str = pub_real_topic
        self._pub_sim_topic: str = pub_sim_topic

    @abstractmethod
    def construct(self) -> None:
        super().construct()

    @abstractmethod
    def publish(self) -> None:
        super().publish()
