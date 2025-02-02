from typing import Optional

from core.types import Config, Mode
from isaacsim import SimulationApp

_universe: Optional["Universe"] = None


def big_bang(
    app_settings: Config,
    world_settings: Config,
    num_envs: int,
    mode: Mode,
    run_name: Optional[str],
    ros_enabled: bool = False,
    experience: str = "./apps/omni.isaac.sim.newton.kit",
) -> "Universe":
    """
    Initializes the universe and the Omniverse simulation app. Can be called multiple times.
    Args:
        app_settings: The settings for the simulation app (i.e. rendering).
        world_settings: The settings for the world (i.e. physics).
        num_envs: The number of environments in the world.
        mode: The mode in which the universe is initialized.
        run_name: The name of the run (if training or playing).
        ros_enabled: Whether to disable ROS2 bridge and ROS-related functionality.
        experience: The path to the experience to load (Omniverse Kit).

    Returns:
        The Universe instance. If called multiple times, returns the same instance.
    """
    global _universe

    if _universe is not None:
        return _universe

    sim_app: SimulationApp = SimulationApp(app_settings, experience)

    if ros_enabled:
        from omni.isaac.core.utils.extensions import enable_extension

        # enable ROS2 bridge extension
        enable_extension("omni.isaac.ros2_bridge")

        import rclpy

        rclpy.init()

    from core.logger import Logger

    Logger.flush()

    from core.universe import Universe

    _universe = Universe(
        headless=app_settings["headless"],
        sim_app=sim_app,
        world_settings=world_settings,
        num_envs=num_envs,
        mode=mode,
        run_name=run_name,
        ros_enabled=ros_enabled,
    )
    _universe.reset(soft=False)

    return _universe
