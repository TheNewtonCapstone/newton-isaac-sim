source _isaac_sim/setup_conda_env.sh

EXTRA_PACKAGE_PATHS=(
    _isaac_sim/kit/exts/omni.usd.libs
    _isaac_sim/exts/omni.isaac.core
    _isaac_sim/exts/omni.isaac.sensor
    _isaac_sim/exts/omni.isaac.cloner
    _isaac_sim/extsPhysics/omni.physx
    _isaac_sim/extsPhysics/omni.usd.schema.physx
)

for path in "${EXTRA_PACKAGE_PATHS[@]}"; do
    EXTRA_PYTHON_PATH="$EXTRA_PYTHON_PATH:$PWD/$path"
done

export PYTHONPATH=$PYTHONPATH:$EXTRA_PYTHON_PATH