#!/bin/bash

# Purpose
# This script is used to build all ROS packages for Newton

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "ROS2 is not sourced. Please source ROS2."
    exit 1
fi


# Clean previous ROS2 packages builds
if [ -d "build" ]; then
    echo "Cleaning previous build files..."
    rm -rf "build" "install" "log"
fi

# Build ROS2 packages
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE:INTERNAL="$(which python3)" -DPython3_FIND_STRATEGY=LOCATION -DPython3_FIND_UNVERSIONED_NAMES=FIRST
