#!/bin/bash

# Purpose
# Setup the environment for ROS2, meant to be ran from this directory (and generally assumes that ROS2 is installed through Robostack)

FASTRTPS_DEFAULT_PROFILES_FILE="$(realpath -s "fastdds.xml")"

export FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE
export ROS_DOMAIN_ID=25 # Arbitrarily chosen

source source_pkgs.sh