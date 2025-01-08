# ROS2 imports
try:
    from .ros_joints_controller import ROSVecJointsController
except ImportError as e:
    print(
        "ROS2 probably not enabled: skipping ROS2 imports for 'controllers' module.", e
    )

# ROS2-less imports
from .joints_controller import VecJointsController
