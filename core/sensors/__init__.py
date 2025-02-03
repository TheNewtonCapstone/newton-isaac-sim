# ROS2 imports
try:
    from .ros_imu import ROSVecIMU
    from .ros_contact import ROSVecContact
except ImportError as e:
    from ..logger import Logger

    Logger.warning(
        f"ROS2 probably not enabled: skipping ROS2 imports for 'sensors' module ({e})"
    )

# ROS2-less imports
from .imu import VecIMU
from .contact import VecContact
