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
    echo "Sourcing package in $dir ..."
    cd "$dir" || exit
    source install/setup.bash
    cd - || exit
done