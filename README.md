# Newton - Isaac Sim
This repo contains the code and configuration used during our journey of developing Newton using NVIDIA's Isaac Sim.

## Requirements
- Ubuntu 22.04 LTS (Works with Pop OS as well)
- Nvidia GPU (RTX 2070 & above) with 510.73.05+ drivers (execute `nvidia-smi` in your terminal to make sure the drivers are set up)
- Isaac Sim (tested with version `4.2.0`)
- Stable Baselines 3
- Anaconda | Miniconda
  - If using ROS: install Mamba (optional; with `conda install mamba -c conda-forge`) for faster package installation

## Isaac Sim Setup
- Download Isaac Sim by following the steps found within Nvidia's installation [guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html).
- Run Isaac sim from the Omniverse application to make sure it runs properly
- Clone this repository `git clone https://github.com/TheNewtonCapstone/newton-isaac-sim`
- Run `cd newton-isaac-sim && ln -s /home/YOUR_USERNAME/.local/share/ov/pkg/isaac-sim-4.2.0 _isaac_sim`
- Create the conda environment `conda env create -f environment.yml`
  - If using ROS: `conda env create -f environment_ros.yml`

## Repo Structure
- `newton.py`: Main script used for training/testing/exporting models
- `environment(_ros).yml`: Project dependency requirements
- `core/`: Contains all the core functionalities (Simulation, Training, Animation...)
- `assets/`: Contains miscellaneous assets (e.g. animations, USD files)
- `docs/`: Contains general project documentation 
- `scripts/`: Contains helper scripts such as the animation keyframe extractor
- `configs/`: Contains the configuration files
- `runs/`: Model checkpoints and summaries will be saved here (by default)
- `apps/`: Contains the Isaac Sim applications configuration files

### Running Isaac Sim
The entry point of our project is `newton.py`. **Before running your IDE** (and in any terminal that you wish to run the project in), you must configure the environment:
- `conda activate isaac-sim`
- `source setup.sh`
  - If using ROS: this script will automatically source the Newton ROS workspace

Now with your environment configured, within the same terminal, you can open your desired IDE:
- `pycharm` for Pycharm (recommended)
- `code` for VS Code

We have provided a simple CLI to allow you to start training your very own Newton. All you have to do is run `newton.py`!
Here is how the first option should look like: 

![docs/assets/mode_select.png](docs/assets/mode_select.png)

### Training
- execute `python newton.py` and select `training`.
- Models are saved as `runs/{TaskName}/nn/{checkpoint_name}.pth`

### Exporting ONNX
- execute`python newton.py` and select `Exporting`.
- Model is exported to `runs/{checkpoint_name}/nn/{task_name}.pth.onnx`
