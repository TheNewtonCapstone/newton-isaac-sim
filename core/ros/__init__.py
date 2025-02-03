try:
    from .base_node import BaseNode
    from .base_sim_real_node import BaseSimRealNode
except ImportError as e:
    from ..logger import Logger

    Logger.warning(
        f"ROS2 probably not enabled: skipping ROS2 imports for 'ros' module ({e})."
    )
