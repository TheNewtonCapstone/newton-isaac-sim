# TWIP - Isaac Sim
This repo contains the code and configuration used during our journey of developing TWIP using NVIDIA's Isaac Sim and rl games.

## Requirements
- Ubuntu 22.04 LTS (Works with Pop OS as well)
 - Nvidia GPU with 510.73.05+ drivers (`nvidia-smi` to make sure these are set up)
- Isaac Sim (tested with version `4.0.0`)
- RL games
- Anaconda | Miniconda

## Isaac Sim Setup
- Download Isaac Sim by following the steps found within Nvidia's installation (guide)[https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html].
- Run Isaac sim from the Omniverse application to make sure it runs properly
- Clone this repository `git clone https://github.com/TheNewtonCapstone/twip-isaac-sim`
- Remove any folder named `_isaac_sim`
- Run `cd twip-isaac-sim && ln -s home/YOUR_NAME/.local/share/ov/pkg/isaac-sim-4.0.0`
- Create the conda environment `conda env create -f environment.yml` 

## Repo Structure
 - `twip.py`: Main script used for training/testing/exporting models
- `export.py`: Script for exporting models as ONNX files
- `cfg/task/`: YAML configs for each task defining environment parameters and domain randomization
- `cfg/train/`: YAML configs for each task defining RL model and hyperparameters
- `tasks/`: Class definitions and assets for each task
- `runs/`: Model checkpoints and summaries will be saved here

## Usage

### Running Isaac Sim
The entry point of our project is `twip.py`. To run our project you must configure the environment:
- `conda activate isaac-sim`
- `source _isaac_sim/setup_conda_env.sh`
- `python twip.py --sim-only --num-envs 4`

`--sim-only` is used to run the simluation alone without any reinforcement learning. `--num-envs` specifies the number of twip environments within the simulation. To see all the options, run `twip.py --help`.

### Training
- `python train.py task={task_name}` to begin training
- Models are saved as `runs/{TaskName}/nn/{checkpoint_name}.pth`

### Exporting ONNX
 - `python twip.py --checkpoint="runs/{checkpoint_name}/nn/{task_name}.pth --export-onnx"` to export
 - Model is exported to `runs/{checkpoint_name}/nn/{task_name}.pth.onnx`

### Adding a New Task (?)
Create a new folder called `tasks/{task_name}`. The following files will be needed:
- `tasks/{task_name}/{task_name}.py`: Class defining the task. Rewards, observations, actions are all defined here.
- `tasks/{task_name}/assets/`: URDF and meshes should be placed here.
- `cfg/task/{TaskName}.yaml`: Config file defining environment parameters and domain randomization
- `cfg/train/{TaskName}PPO.yaml`: Config file defining hyperparameters for training
