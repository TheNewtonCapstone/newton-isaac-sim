#!/bin/bash

# Sets up the environment for running Isaac Sim in a Conda environment
# This should be used instead of setup_conda_env.sh, because it fixes the PYTHONPATH & tailors it to our needs
# This script should be sourced, not executed

# By default, we source ROS2's setup.bash, but it can be skipped by giving the argument "no-ros2"
if [ "$1" != "no-ros2" ]; then
  # Setup any ROS' environment

  cd ros || exit
  source setup.sh
  cd .. || exit
fi

# Then setup Isaac Sim's environment

ISAAC_SIM_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/_isaac_sim"
MY_DIR="$(realpath -s "$ISAAC_SIM_DIR")"

export CARB_APP_PATH=$ISAAC_SIM_DIR/kit
export EXP_PATH=$MY_DIR/apps
export ISAAC_PATH=$MY_DIR

EXTRA_LD_LIBRARY_PATHS=(
    "$ISAAC_SIM_DIR/."
    "$ISAAC_SIM_DIR/exts/omni.usd.schema.isaac/plugins/IsaacSensorSchema/lib"
    "$ISAAC_SIM_DIR/exts/omni.usd.schema.isaac/plugins/RangeSensorSchema/lib"
    "$ISAAC_SIM_DIR/exts/omni.isaac.lula/pip_prebundle"
    "$ISAAC_SIM_DIR/exts/omni.exporter.urdf/pip_prebundle"
    "$ISAAC_SIM_DIR/exts/omni.isaac.ros2_bridge/humble/lib"
    "$ISAAC_SIM_DIR/kit"
    "$ISAAC_SIM_DIR/kit/kernel/plugins"
    "$ISAAC_SIM_DIR/kit/libs/iray"
    "$ISAAC_SIM_DIR/kit/plugins"
    "$ISAAC_SIM_DIR/kit/plugins/bindings-python"
    "$ISAAC_SIM_DIR/kit/plugins/carb_gfx"
    "$ISAAC_SIM_DIR/kit/plugins/rtx"
    "$ISAAC_SIM_DIR/kit/plugins/gpu.foundation"
)

EXTRA_PYTHON_PATHS=(
    "$ISAAC_SIM_DIR/python_packages"
    "$ISAAC_SIM_DIR/kit/python/lib/python3.10/site-packages"
    "$ISAAC_SIM_DIR/kit/kernel/py"
    "$ISAAC_SIM_DIR/kit/plugins/bindings-python"
    "$ISAAC_SIM_DIR/kit/exts/omni.kit.pip_archive/pip_prebundle"
    "$ISAAC_SIM_DIR/kit/exts/omni.usd.libs"
    "$ISAAC_SIM_DIR/exts/omni.isaac.lula/pip_prebundle"
    "$ISAAC_SIM_DIR/exts/omni.exporter.urdf/pip_prebundle"
    "$ISAAC_SIM_DIR/exts/omni.isaac.core_archive/pip_prebundle"
    "$ISAAC_SIM_DIR/exts/omni.isaac.ml_archive/pip_prebundle"
    "$ISAAC_SIM_DIR/exts/omni.pip.compute/pip_prebundle"
    "$ISAAC_SIM_DIR/exts/omni.pip.cloud/pip_prebundle"
    "$ISAAC_SIM_DIR/exts/omni.isaac.kit"
    "$ISAAC_SIM_DIR/exts/omni.isaac.gym"
    "$ISAAC_SIM_DIR/exts/omni.isaac.core"
    "$ISAAC_SIM_DIR/exts/omni.isaac.sensor"
    "$ISAAC_SIM_DIR/exts/omni.isaac.cloner"
    "$ISAAC_SIM_DIR/extsPhysics/omni.physx"
    "$ISAAC_SIM_DIR/extsPhysics/omni.physics.tensors"
    "$ISAAC_SIM_DIR/extsPhysics/omni.usd.schema.physx"
)

for path in "${EXTRA_LD_LIBRARY_PATHS[@]}"; do
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$path
done

for path in "${EXTRA_PYTHON_PATHS[@]}"; do
    export PYTHONPATH=$PYTHONPATH:$path
done