#!/bin/bash

# Purpose
# This script is used to build all ROS packages for Newton

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "ROS2 is not sourced. Please source ROS2."
    exit 1
fi

# Go in each package directory (within the "ros" directory) and build it
shopt -s nullglob

for dir in ./*/; do
    cd "$dir" || exit

    if [ -d "build" ]; then
        echo "Cleaning previous build files in $dir ..."
        rm -rf "build" "install" "log"
    fi

    echo "Building package in $dir ..."
    colcon build --cmake-args -DPython3_EXECUTABLE:INTERNAL="$(which python3)" -DPython3_FIND_STRATEGY=LOCATION -DPython3_FIND_UNVERSIONED_NAMES=FIRST
    cd - || exit
done