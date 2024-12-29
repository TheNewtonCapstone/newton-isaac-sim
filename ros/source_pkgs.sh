#!/bin/bash

# Purpose
# This script is used to source all ROS packages for Newton

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "ROS2 is not sourced. Please source ROS2."
    exit 1
fi

# Source ROS2 packages
source install/local_setup.bash