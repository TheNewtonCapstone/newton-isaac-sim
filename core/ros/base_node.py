from abc import abstractmethod
from typing import Type, List, Dict

from rclpy.node import Node
from rclpy.qos import QoSProfile

from ..universe import Universe


class BaseNode(Node):
    def __init__(
        self,
        universe: Universe,
        node_name: str,
        pub_msg_types: List[Type],
        pub_topics: List[str],
        namespace: str,
        pub_period: int,
        pub_qos_profile: QoSProfile,
    ):
        Node.__init__(
            self,
            node_name=node_name,
            namespace=namespace,
        )

        self._universe: Universe = universe
        self._time: int = 0

        self._pub_msg_types: List[Type] = pub_msg_types
        self._pub_topics: List[str] = pub_topics
        self._pub_period: int = pub_period
        self._pub_qos_profile: QoSProfile = pub_qos_profile

        from rclpy.publisher import Publisher

        self._indexed_publishers: Dict[str, Publisher] = {}

        self._is_node_constructed: bool = False

    @property
    def pub_period(self) -> int:
        return self._pub_period

    @property
    def time(self) -> int:
        """Time counted by the number of times the step method was called. Depending on the overall context,
        this might be in sync with the physics step, rl step or something else entirely.
        """
        return self._time

    @abstractmethod
    def construct(self) -> None:
        assert (
            not self._is_node_constructed
        ), f"BaseNode (from {self.__class__.__name__}) is already constructed: tried to construct!"

        for i, topic in enumerate(self._pub_topics):
            pub = self.create_publisher(
                self._pub_msg_types[i],
                topic,
                qos_profile=self._pub_qos_profile,
            )

            self._indexed_publishers[topic] = pub

        self._is_node_constructed = True

    def step(self) -> None:
        assert (
            self._is_node_constructed
        ), f"BaseNode (from {self.__class__.__name__}) is not constructed: tried to step!"

        self._time += 1

        if self.time % self.pub_period == 0:
            self.publish()

    @abstractmethod
    def publish(self) -> None:
        assert (
            self._is_node_constructed
        ), f"BaseNode (from {self.__class__.__name__}) is not constructed: tried to publish!"
