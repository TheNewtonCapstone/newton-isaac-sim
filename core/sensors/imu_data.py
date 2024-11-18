from dataclasses import dataclass

import torch


@dataclass
class IMUData:
    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where ``N`` is the number of sensors.
    """

    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion ``(w, x, y, z)`` in world frame.

    Shape is (N, 4), where ``N`` is the number of sensors.
    """

    ang_vel_b: torch.Tensor = None
    """Root angular velocity in body frame.

    Shape is (N, 3), where ``N`` is the number of sensors.
    """

    lin_acc_b: torch.Tensor = None
    """Root linear acceleration in body frame.

    Shape is (N, 3), where ``N`` is the number of sensors.
    """