import abc
from enum import Enum
import numpy as np
import torch

from core.base.base_agent import BaseAgent

from typing import Dict



# when implementing ROS, check the following link: https://github.com/ros2/examples/blob/rolling/rclpy/topics/minimal_publisher
# probably a good idea to make separate wrapper classes for each joint (or rather, for each object that will publish/subscribe to ROS messages)


# this class describes how the agent will be constructed, nothing more
class TwipAgent(BaseAgent):
    def __init__(self, config) -> None:
        super().__init__(config)

    def construct(self, root_path: str, world) -> bool:
        super().construct(root_path, world)

        twip_prim_path = root_path + "/twip"

        # these only work after SimulationApp is initialized (to be done in scripts that import this class)
        import omni.isaac.core.utils.stage as stage_utils

        stage_utils.add_reference_to_stage(
            self.config["twip_usd_path"], prim_path=twip_prim_path
        )  # /envs/e_0/twip

        #from omni.isaac.sensor import IMUSensor

        world.reset()

        #self.imu: IMUSensor = world.scene.add(
        #    IMUSensor(
        #        prim_path=twip_prim_path + "/body/imu",
        #        name="imu",
        #    )
        #)
