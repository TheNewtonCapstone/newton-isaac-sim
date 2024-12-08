#!/bin/bash

# Sets up the environment for running Isaac Sim in a Conda environment
# This should be used instead of setup_conda_env.sh, because it fixes the PYTHONPATH & tailors it to our needs

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/_isaac_sim"
MY_DIR="$(realpath -s "$SCRIPT_DIR")"

export CARB_APP_PATH=$SCRIPT_DIR/kit
export EXP_PATH=$MY_DIR/apps
export ISAAC_PATH=$MY_DIR

EXTRA_LD_LIBRARY_PATHS=(
    "$SCRIPT_DIR/."
    "$SCRIPT_DIR/exts/omni.usd.schema.isaac/plugins/IsaacSensorSchema/lib"
    "$SCRIPT_DIR/exts/omni.usd.schema.isaac/plugins/RangeSensorSchema/lib"
    "$SCRIPT_DIR/exts/omni.isaac.lula/pip_prebundle"
    "$SCRIPT_DIR/exts/omni.exporter.urdf/pip_prebundle"
    "$SCRIPT_DIR/kit"
    "$SCRIPT_DIR/kit/kernel/plugins"
    "$SCRIPT_DIR/kit/libs/iray"
    "$SCRIPT_DIR/kit/plugins"
    "$SCRIPT_DIR/kit/plugins/bindings-python"
    "$SCRIPT_DIR/kit/plugins/carb_gfx"
    "$SCRIPT_DIR/kit/plugins/rtx"
    "$SCRIPT_DIR/kit/plugins/gpu.foundation"
)

EXTRA_PACKAGE_PATHS=(
    "$SCRIPT_DIR/python_packages"
    "$SCRIPT_DIR/kit/python/lib/python3.10/site-packages"
    "$SCRIPT_DIR/kit/kernel/py"
    "$SCRIPT_DIR/kit/plugins/bindings-python"
    "$SCRIPT_DIR/kit/exts/omni.kit.pip_archive/pip_prebundle"
    "$SCRIPT_DIR/kit/exts/omni.usd.libs"
    "$SCRIPT_DIR/exts/omni.isaac.lula/pip_prebundle"
    "$SCRIPT_DIR/exts/omni.exporter.urdf/pip_prebundle"
    "$SCRIPT_DIR/exts/omni.isaac.core_archive/pip_prebundle"
    "$SCRIPT_DIR/exts/omni.isaac.ml_archive/pip_prebundle"
    "$SCRIPT_DIR/exts/omni.pip.compute/pip_prebundle"
    "$SCRIPT_DIR/exts/omni.pip.cloud/pip_prebundle"
    "$SCRIPT_DIR/exts/omni.isaac.kit"
    "$SCRIPT_DIR/exts/omni.isaac.gym"
    "$SCRIPT_DIR/exts/omni.isaac.core"
    "$SCRIPT_DIR/exts/omni.isaac.sensor"
    "$SCRIPT_DIR/exts/omni.isaac.cloner"
    "$SCRIPT_DIR/extsPhysics/omni.physx"
    "$SCRIPT_DIR/extsPhysics/omni.physics.tensors"
    "$SCRIPT_DIR/extsPhysics/omni.usd.schema.physx"
)

for path in "${EXTRA_LD_LIBRARY_PATHS[@]}"; do
    EXTRA_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$path
done

for path in "${EXTRA_PACKAGE_PATHS[@]}"; do
    EXTRA_PYTHON_PATH=$EXTRA_PYTHON_PATH:$path
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$EXTRA_LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$EXTRA_PYTHON_PATH