from rclpy.qos import QoSProfile

from core.types import Config


def is_valid_duration_config(duration_config: Config) -> bool:
    """
    Check if the given duration config is valid.
    Args:
        duration_config: The duration config.

    Returns:
        Whether the duration config is valid.
    """
    return "seconds" in duration_config and "nanoseconds" in duration_config


def get_default_qos_profile_from_config(ros_config: Config) -> QoSProfile:
    """
    Get the default QoS profile from the ROS config.
    Args:
        ros_config: The ROS config.

    Returns:
        The default QoS profile.
    """
    default_qos_config: Config = ros_config["defaults"]["qos"]

    from rclpy.qos import (
        HistoryPolicy,
        ReliabilityPolicy,
        DurabilityPolicy,
        LivelinessPolicy,
    )
    from rclpy.duration import Duration

    assert is_valid_duration_config(
        default_qos_config["deadline"]
    ), "Invalid default deadline config!"
    assert is_valid_duration_config(
        default_qos_config["lifespan"]
    ), "Invalid default lifespan config!"
    assert is_valid_duration_config(
        default_qos_config["liveliness_lease_duration"]
    ), "Invalid default liveliness lease duration config!"

    history: HistoryPolicy = default_qos_config["history"]
    depth: int = default_qos_config["depth"]
    reliability: ReliabilityPolicy = default_qos_config["reliability"]
    durability: DurabilityPolicy = default_qos_config["durability"]
    deadline: Duration = Duration(
        seconds=default_qos_config["deadline"]["seconds"],
        nanoseconds=default_qos_config["deadline"]["nanoseconds"],
    )
    lifespan: Duration = Duration(
        seconds=default_qos_config["lifespan"]["seconds"],
        nanoseconds=default_qos_config["lifespan"]["nanoseconds"],
    )
    liveliness: LivelinessPolicy = default_qos_config["liveliness"]
    liveliness_lease_duration: Duration = Duration(
        seconds=default_qos_config["liveliness_lease_duration"]["seconds"],
        nanoseconds=default_qos_config["liveliness_lease_duration"]["nanoseconds"],
    )

    return QoSProfile(
        history=history,
        depth=depth,
        reliability=reliability,
        durability=durability,
        deadline=deadline,
        lifespan=lifespan,
        liveliness=liveliness,
        liveliness_lease_duration=liveliness_lease_duration,
    )


def get_qos_profile_from_node_config(
    node_config: Config,
    qos_config_key: str,
    ros_config: Config,
) -> QoSProfile:
    """
    Get the QoS profile from the node config.
    Args:
        node_config: The node config.
        qos_config_key: The key in the node config that contains the QoS config.
        ros_config: The ROS config.

    Returns:
        The QoS profile.
    """
    default_qos_profile = get_default_qos_profile_from_config(ros_config)

    assert (
        qos_config_key in node_config
    ), f"Given QoS config key '{qos_config_key}' was not found in node config!"

    node_qos_config = node_config[qos_config_key]

    # maybe the node doesn't have a specific QoS config, so the default one is used
    if node_qos_config is None:
        return default_qos_profile

    from rclpy.qos import (
        HistoryPolicy,
        ReliabilityPolicy,
        DurabilityPolicy,
        LivelinessPolicy,
    )
    from rclpy.duration import Duration

    history: HistoryPolicy = node_qos_config.get(
        "history",
        default_qos_profile.history,
    )
    depth: int = node_qos_config.get(
        "depth",
        default_qos_profile.depth,
    )
    reliability: ReliabilityPolicy = node_qos_config.get(
        "reliability",
        default_qos_profile.reliability,
    )
    durability: DurabilityPolicy = node_qos_config.get(
        "durability",
        default_qos_profile.durability,
    )
    deadline: Duration = (
        Duration(
            seconds=node_qos_config["deadline"]["seconds"],
            nanoseconds=node_qos_config["deadline"]["nanoseconds"],
        )
        if is_valid_duration_config(node_qos_config["deadline"])
        else default_qos_profile.deadline
    )
    lifespan: Duration = (
        Duration(
            seconds=node_qos_config["lifespan"]["seconds"],
            nanoseconds=node_qos_config["lifespan"]["nanoseconds"],
        )
        if is_valid_duration_config(node_qos_config["lifespan"])
        else default_qos_profile.lifespan
    )
    liveliness: LivelinessPolicy = node_qos_config.get(
        "liveliness",
        default_qos_profile.liveliness,
    )
    liveliness_lease_duration: Duration = (
        Duration(
            seconds=node_qos_config["liveliness_lease_duration"]["seconds"],
            nanoseconds=node_qos_config["liveliness_lease_duration"]["nanoseconds"],
        )
        if is_valid_duration_config(node_qos_config["liveliness_lease_duration"])
        else default_qos_profile.liveliness_lease_duration
    )

    return QoSProfile(
        history=history,
        depth=depth,
        reliability=reliability,
        durability=durability,
        deadline=deadline,
        lifespan=lifespan,
        liveliness=liveliness,
        liveliness_lease_duration=liveliness_lease_duration,
    )
