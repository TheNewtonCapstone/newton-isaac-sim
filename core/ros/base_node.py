from abc import abstractmethod
from typing import Type, Optional

from rclpy.node import Node
from rclpy.qos import QoSProfile

from ..universe import Universe


class BaseNode(Node):
    def __init__(
        self,
        universe: Universe,
        publishing_msg_type: Type,
        node_name: str,
        publishing_topic: str,
        namespace: str = "sim",
        publishing_frequency: float = 5.0,
        publishing_qos_profile: QoSProfile = QoSProfile(depth=0),
    ):
        Node.__init__(
            self,
            node_name=node_name,
            namespace=namespace,
        )

        self._universe: Universe = universe
        self._time: int = 0

        self._publishing_msg_type = publishing_msg_type
        self._publishing_topic = publishing_topic
        self._publishing_frequency = publishing_frequency
        self._publishing_qos_profile = publishing_qos_profile

        from rclpy.publisher import Publisher

        self._publisher: Optional[Publisher] = None

        self._is_node_constructed = False

    @property
    def publishing_frequency(self) -> float:
        return self._publishing_frequency

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

        self._publisher = self.create_publisher(
            self._publishing_msg_type,
            self._publishing_topic,
            qos_profile=self._publishing_qos_profile,
        )

        self._is_node_constructed = True

    def step(self) -> None:
        assert (
            self._is_node_constructed
        ), f"BaseNode (from {self.__class__.__name__}) is not constructed: tried to step!"

        self._time += 1

        if self.time % self.publishing_frequency == 0:
            self.publish()

    @abstractmethod
    def publish(self) -> None:
        assert (
            self._is_node_constructed
        ), f"BaseNode (from {self.__class__.__name__}) is not constructed: tried to publish!"
