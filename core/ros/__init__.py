try:
    from .base_node import BaseNode
except ImportError as e:
    print("ROS2 probably not enabled: skipping ROS2 imports for 'ros'.", e)
