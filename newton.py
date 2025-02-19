import argparse
import os
from typing import List, Optional, Tuple, get_args

from core.logger import Logger
from core.types import Matter, Config, ConfigCollection, Mode


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newton.py",
        description="Entrypoint for any Newton-related actions.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode.",
        default=False,
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        help="Path to the configuration file for the current robot.",
        default="configs/robots/newton.yaml",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        help="Path to the directory containing the tasks.",
        default="configs/tasks",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        help="Name of the task to load (for RL).",
        default=None,
    )
    parser.add_argument(
        "--network-config",
        type=str,
        help="Path to the network configuration file for RL.",
        default="configs/networks.yaml",
    )
    parser.add_argument(
        "--network-name",
        type=str,
        help="Name of the network configuration to be used (located in the network configration file).",
        default=None,
    )
    parser.add_argument(
        "--world-config",
        type=str,
        help="Path to the configuration file for the world.",
        default="configs/world.yaml",
    )
    parser.add_argument(
        "--randomization-config",
        type=str,
        help="Path to the configuration file for domain randomization.",
        default="configs/randomization.yaml",
    )
    parser.add_argument(
        "--ros-config",
        type=str,
        help="Path to the configuration file for ROS2.",
        default="configs/ros.yaml",
    )
    parser.add_argument(
        "--db-config",
        type=str,
        help="Path to the configuration file for the database integration.",
        default="configs/db.yaml",
    )
    parser.add_argument(
        "--logger-config",
        type=str,
        help="Path to the configuration file for the logger.",
        default="configs/logger.yaml",
    )
    parser.add_argument(
        "--secrets",
        type=str,
        help="Path to the secrets file (containing database token, API keys, etc.).",
        default="configs/secrets.yaml",
    )
    parser.add_argument(
        "--animations-dir",
        type=str,
        help="Path to the directory containing the animations.",
        default="assets/newton/animations",
    )
    parser.add_argument(
        "--animation-name",
        type=str,
        help="Name of the animation to load (for RL or animating).",
        default=None,
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        help="Path to the directory containing checkpoints.",
        default="runs",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to the checkpoint file to load (for RL or exporting).",
        default=None,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name of the run to load (for RL or exporting; will select highest reward checkpoint within).",
        default=None,
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Skip any checkpoint loading logic (will fail when playing or exporting).",
        default=False,
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Number of environments to run (will be read from the rl-config if not specified).",
        default=-1,
    )
    parser.add_argument(
        "--terrain-config",
        type=str,
        help="Path to the configuration file for the terrain.",
        default="configs/terrain.yaml",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the agent using an optional checkpoint.",
        default=False,
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the agent using a trained checkpoint.",
        default=False,
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate the agent using a parsed animation file.",
        default=False,
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Run the simulation only (no RL).",
        default=False,
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Exports checkpoint as ONNX model.",
        default=False,
    )

    return parser


def mode_select(modes: List[str]) -> Tuple[str, int]:
    from bullet import Bullet

    cli = Bullet(
        prompt="Please select a mode:",
        choices=modes,
        align=2,
        margin=1,
        return_index=True,
    )

    return cli.launch()


def task_select(
    task_config_dir: str,
    task_name: Optional[str],
) -> Optional[Tuple[ConfigCollection, Config, str]]:
    from core.utils.config import load_named_configs_in_dir

    # discover tasks in directory
    task_configs = load_named_configs_in_dir(task_config_dir)

    if len(task_configs.items()) == 0:
        Logger.error(f"No task files found in {task_config_dir}.")
        return None

    # load animation clips settings from config files
    task_names: List[str] = list(task_configs.keys())

    # use the specified animation if it exists
    if task_name and task_name in task_configs:
        return task_configs, task_configs[task_name], task_name

    from bullet import Bullet

    # otherwise, ask the user to manually select an animation
    cli = Bullet(
        prompt="Please select a task:",
        choices=task_names,
        align=2,
        margin=1,
        return_index=True,
    )

    selected_task_name, selected_task_idx = cli.launch()

    return (
        task_configs,
        list(task_configs.values())[selected_task_idx],
        list(task_configs.values())[selected_task_idx]["name"],
    )


def animation_select(
    animation_config_dir: str,
    current_animation_name: Optional[str],
) -> Optional[Tuple[ConfigCollection, Config, str]]:
    from core.utils.config import load_named_configs_in_dir

    # discover animations in directory
    animation_clips_configs = load_named_configs_in_dir(animation_config_dir)

    if len(animation_clips_configs.items()) == 0:
        Logger.info(f"No animation files found in {animation_config_dir}.")
        return None

    animation_clips_names: List[str] = list(animation_clips_configs.keys())

    # use the specified animation if it exists
    if current_animation_name and current_animation_name in animation_clips_configs:
        return (
            animation_clips_configs,
            animation_clips_configs[current_animation_name],
            current_animation_name,
        )

    from bullet import Bullet

    # otherwise, ask the user to manually select an animation
    cli = Bullet(
        prompt="Please select an animation:",
        choices=animation_clips_names,
        align=2,
        margin=1,
        return_index=True,
    )

    selected_animation_name, selected_animation_idx = cli.launch()

    return (
        animation_clips_configs,
        list(animation_clips_configs.values())[selected_animation_idx],
        list(animation_clips_configs.values())[selected_animation_idx]["name"],
    )


def checkpoint_select(
    runs_dir: str,
    current_run_name: Optional[str],
    required: bool,
) -> Optional[Tuple[Config, str, str]]:
    from core.utils.runs import (
        save_runs_library,
        build_runs_library_from_runs_folder,
    )

    runs_settings: Config = build_runs_library_from_runs_folder(runs_dir)
    if not runs_settings:
        Logger.error(f"No runs found in {runs_dir}.")
        return None

    save_runs_library(runs_settings, runs_dir)

    if current_run_name and current_run_name in runs_settings:
        if runs_settings[current_run_name]["count"] == 0:
            Logger.error(f"No checkpoints found for run {current_run_name}.")
            return None

        return (
            runs_settings,
            current_run_name,
            runs_settings[current_run_name]["checkpoints"][0]["path"],
        )

    from bullet import Input, YesNo

    if not required:
        cli = YesNo(
            prompt="Would you like to select a checkpoint? ",
            default="n",
        )

        if not cli.launch():
            return None

    cli = Input(
        prompt=f"Please enter a run name (format: <task_name>_001): ",
        default=f"",
        strip=True,
    )

    selected_run_name = ""

    while selected_run_name not in runs_settings:
        selected_run_name = cli.launch()

    if runs_settings[selected_run_name]["count"] == 0:
        Logger.error(f"No checkpoints found for run {selected_run_name}.")
        return None

    return (
        runs_settings,
        selected_run_name,
        runs_settings[selected_run_name]["checkpoints"][0]["path"],
    )


def network_arch_select(
    network_config_file_path: str,
    network_name: Optional[str],
) -> Optional[Tuple[Config, str]]:
    from bullet import Bullet

    from core.utils.config import load_config

    network_config = load_config(network_config_file_path)

    networks: Config = network_config["networks"]

    # gets the direct torch.nn reference from the string
    def get_torch_activation_fn(activation_fn: str):
        import torch.nn

        nn_module_name = activation_fn.split(".")[-1]

        return getattr(torch.nn, nn_module_name)

    if network_name in networks:
        return (
            {
                "net_arch": networks[network_name]["net_arch"],
                "activation_fn": get_torch_activation_fn(
                    networks[network_name]["activation_fn"]
                ),
            },
            network_name,
        )

    # Build choices for CLI
    choices = [
        f"{name}: Network Architecture: {config['net_arch']}, Activation function: {config['activation_fn']}"
        for name, config in networks.items()
    ]

    # CLI for network selection
    cli = Bullet(
        prompt="Select a network configuration:",
        choices=choices,
        align=2,
        margin=1,
        return_index=True,
    )

    selected_choice, selected_choice_ind = cli.launch()

    # Parse selected network
    selected_config: Config = list(networks.values())[selected_choice_ind]

    selected_config["activation_fn"] = get_torch_activation_fn(
        selected_config["activation_fn"]
    )

    return {
        "net_arch": selected_config["net_arch"],
        "activation_fn": selected_config["activation_fn"],
    }, network_name


def setup_logging() -> None:
    from core.utils.config import load_config

    parser = setup_argparser()

    cli_args, _ = parser.parse_known_args()

    logger_config = load_config(cli_args.logger_config)
    log_file_path = "logs/newton.log"

    # creates a singleton instance of the logger, to be used throughout the program
    Logger.create(logger_config, log_file_path)


def setup() -> Optional[Matter]:
    from core.utils.config import load_config

    parser = setup_argparser()

    # General Configs
    cli_args, _ = parser.parse_known_args()

    robot_config = load_config(cli_args.robot_config)
    logger_config = load_config(cli_args.logger_config)
    world_config = load_config(cli_args.world_config)
    randomization_config = load_config(cli_args.randomization_config)
    ros_config = load_config(cli_args.ros_config)
    db_config = load_config(cli_args.db_config)
    secrets = load_config(cli_args.secrets)
    terrain_config = load_config(cli_args.terrain_config)

    # Control Flags
    training = cli_args.train
    playing = cli_args.play
    animating = cli_args.animate
    physics_only = cli_args.physics
    exporting = cli_args.export_onnx

    modes = {
        "Training": training,
        "Playing": playing,
        "Animating": animating,
        "Physics Only": physics_only,
        "Exporting": exporting,
    }
    mode: Mode

    none_selected = sum(modes.values()) == 0
    multiple_selected = sum(modes.values()) > 1

    if multiple_selected:
        Logger.error(
            "Only one of training, playing, animating, physics or exporting can be set."
        )
        return None
    elif (
        none_selected
    ):  # we need a mode to start, so we ask the user to select one interactively
        mode_name, mode_idx = mode_select(list(modes.keys()))

        training = mode_idx == 0
        playing = mode_idx == 1
        animating = mode_idx == 2
        physics_only = mode_idx == 3
        exporting = mode_idx == 4

        mode = get_args(Mode)[mode_idx]
    else:  # we have a mode passed by argument
        modes_list = list(modes.values())
        modes_flag_idx = modes_list.index(True)

        mode = get_args(Mode)[modes_flag_idx]
        mode_name = list(modes.keys())[modes_flag_idx]

    # Run & checkpoint config

    runs_library: Config = {}
    runs_dir: str = cli_args.runs_dir

    current_checkpoint_path: Optional[str] = cli_args.checkpoint_path
    current_run_name: Optional[str] = cli_args.run_name
    no_checkpoint = cli_args.no_checkpoint

    if (playing or exporting) and no_checkpoint:
        Logger.error("No checkpoint specified for playing or exporting.")
        return None

    if (
        (training or playing or exporting)
        and current_checkpoint_path is None
        and not no_checkpoint
    ):
        checkpoint_select_result = checkpoint_select(
            runs_dir,
            current_run_name,
            required=not training,
        )

        if checkpoint_select_result is None and not training:
            return None

        if checkpoint_select_result is not None:
            runs_library, current_run_name, current_checkpoint_path = (
                checkpoint_select_result
            )

    # Task

    task_configs: ConfigCollection = {}
    current_task_config: Config = {}
    current_task: Optional[str] = None

    if training and current_run_name is None:
        task_select_result = task_select(
            cli_args.tasks_dir,
            cli_args.task_name,
        )

        # We failed to find an animation, exit
        if task_select_result is None:
            Logger.error("No task found.")
            return None

        task_config, current_task_config, current_task = task_select_result
    elif playing or exporting or (training and current_run_name is not None):
        recorded_task_config_path = os.path.join(
            runs_dir,
            current_run_name,
            "records",
            "task_config_record.yaml",
        )
        current_task_config = load_config(recorded_task_config_path)
        current_task = current_task_config["name"]

    # Animation

    animation_clips_config: ConfigCollection = {}
    current_animation_clip_config: Config = {}
    current_animation: Optional[str] = None

    if animating:
        animation_select_result = animation_select(
            cli_args.animations_dir,
            cli_args.animation_name,
        )

        # We failed to find an animation, and we can't proceed, exit
        if animation_select_result is None and animating:
            return None

        animation_clips_config, current_animation_clip_config, current_animation = (
            animation_select_result
        )
    elif training:
        from core.utils.config import load_named_configs_in_dir

        animation_clips_config = load_named_configs_in_dir(cli_args.animations_dir)
        current_animation_clip_config = animation_clips_config[current_task]
        current_animation = current_animation_clip_config["name"]
    elif playing:
        recorded_animation_clip_config_path = os.path.join(
            runs_dir,
            current_run_name,
            "records",
            "animation_clip_config_record.yaml",
        )
        current_animation_clip_config = load_config(recorded_animation_clip_config_path)
        current_animation = current_animation_clip_config["name"]

    # if we're training and the user did specify no-checkpoint, we need to create a new run
    if no_checkpoint and training and current_run_name is None:
        from core.utils.runs import (
            get_unused_run_id,
            create_runs_library,
        )

        new_run_id = get_unused_run_id(runs_library, current_task)

        if new_run_id is None:
            runs_library = create_runs_library(runs_dir)

            new_run_id = get_unused_run_id(runs_library, current_task)

            if new_run_id is None:
                new_run_id = 0

        current_run_name = f"{current_task}_{new_run_id:03}"

    log_file_path = f"logs/{mode_name.lower().replace(' ', '_')}.log"
    if current_run_name:
        log_file_path = f"{runs_dir}/{current_run_name}/{current_run_name}.log"

    # Network config

    network_config: Config = {}
    network_name: Optional[str] = cli_args.network_name

    if training:
        network_select_result = network_arch_select(
            cli_args.network_config,
            network_name,
        )

        if network_select_result is None:
            Logger.error("No network architecture found.")
            return None

        network_config, network_name = network_select_result
    elif playing or exporting:
        recorded_network_config_path = os.path.join(
            runs_dir,
            current_run_name,
            "records",
            "network_config_record.yaml",
        )
        network_config = load_config(recorded_network_config_path)
        network_name = network_config["name"]

    # Helper flags

    is_rl = training or playing
    headless = (
        cli_args.headless or cli_args.export_onnx
    )  # if we're exporting, don't show the GUI
    interactive = not headless and (
        physics_only or playing or animating
    )  # interactive means that the user is expected to control the agent in some way
    enable_ros = ros_config["enabled"]
    enable_db = db_config["enabled"]

    # override some config with CLI num_envs, if specified (or when not training/playing, we default to 1)
    num_envs = current_task_config["n_envs"] if is_rl else 1
    if cli_args.num_envs != -1:
        num_envs = cli_args.num_envs

    if exporting:
        headless = True

    if interactive:
        world_config["sim_params"]["enable_scene_query_support"] = True

    control_step_dt = world_config["control_dt"]
    inverse_control_frequency = int(control_step_dt / world_config["physics_dt"])

    matter = {
        "cli_args": cli_args,
        "robot_config": robot_config,
        "task_configs": task_configs,
        "current_task_config": current_task_config,
        "current_task_name": current_task,
        "world_config": world_config,
        "randomization_config": randomization_config,
        "network_config": network_config,
        "network_name": network_name,
        "ros_config": ros_config,
        "db_config": db_config,
        "logger_config": logger_config,
        "log_file_path": log_file_path,
        "terrain_config": terrain_config,
        "secrets": secrets,
        "animation_clips_config": animation_clips_config,
        "current_animation_clip_config": current_animation_clip_config,
        "current_animation": current_animation,
        "runs_dir": runs_dir,
        "runs_library": runs_library,
        "current_run_name": current_run_name,
        "current_checkpoint_path": current_checkpoint_path,
        "mode": mode,
        "mode_name": mode_name,
        "training": training,
        "playing": playing,
        "animating": animating,
        "physics_only": physics_only,
        "exporting": exporting,
        "is_rl": is_rl,
        "interactive": interactive,
        "headless": headless,
        "enable_ros": enable_ros,
        "enable_db": enable_db,
        "num_envs": num_envs,
        "control_step_dt": control_step_dt,
        "inverse_control_frequency": inverse_control_frequency,
    }

    return matter


def main():
    setup_logging()

    try:
        base_matter = setup()
    except KeyboardInterrupt:
        Logger.info("Exiting...")
        return
    except Exception as e:
        Logger.fatal(f"An error occurred during setup: {e}")
        return

    if base_matter is None:
        return

    robot_config = base_matter["robot_config"]
    task_configs = base_matter["task_configs"]
    current_task_config = base_matter["current_task_config"]
    current_task_name = base_matter["current_task_name"]
    world_config = base_matter["world_config"]
    randomization_config = base_matter["randomization_config"]
    network_config = base_matter["network_config"]
    ros_config = base_matter["ros_config"]
    db_config = base_matter["db_config"]
    logger_config = base_matter["logger_config"]
    log_file_path = base_matter["log_file_path"]
    terrain_config = base_matter["terrain_config"]
    secrets = base_matter["secrets"]
    animation_clips_config = base_matter["animation_clips_config"]
    current_animation_clip_config = base_matter["current_animation_clip_config"]
    current_animation = base_matter["current_animation"]
    runs_dir = base_matter["runs_dir"]
    current_run_name = base_matter["current_run_name"]
    current_checkpoint_path = base_matter["current_checkpoint_path"]
    mode = base_matter["mode"]
    mode_name = base_matter["mode_name"]
    training = base_matter["training"]
    playing = base_matter["playing"]
    animating = base_matter["animating"]
    physics_only = base_matter["physics_only"]
    exporting = base_matter["exporting"]
    is_rl = base_matter["is_rl"]
    interactive = base_matter["interactive"]
    headless = base_matter["headless"]
    enable_ros = base_matter["enable_ros"]
    enable_db = base_matter["enable_db"]
    num_envs = base_matter["num_envs"]
    control_step_dt = base_matter["control_step_dt"]
    inverse_control_frequency = base_matter["inverse_control_frequency"]

    # set the log file path to the one we've determined the specific one, not the generic one
    Logger.set_log_file_path(log_file_path)

    if is_rl:
        Logger.info(
            f"Running with {num_envs} environments, {current_task_config['ppo']['n_steps']} steps per environment, ROS {'enabled' if enable_ros else 'disabled'} and {'headless' if headless else 'GUI'} mode.\n"
            f"{mode_name}{(' (with checkpoint ' + current_checkpoint_path + ')') if current_checkpoint_path is not None else ''}.\n"
            f"Using {current_task_config['device']} as the RL device and {world_config['device']} as the physics device.",
        )
    else:
        Logger.info(
            f"Running with {num_envs} environments, ROS {'enabled' if enable_ros else 'disabled'} and {'headless' if headless else 'GUI'} mode.\n"
            f"{mode_name}.\n"
            f"Using {world_config['device']} as the physics device.",
        )

    import torch

    # big_bang must be imported & invoked first, to load all necessary omniverse extensions
    from core import big_bang

    universe = big_bang(
        {"headless": headless},
        world_config,
        num_envs=num_envs,
        mode=mode,
        run_name=current_run_name,
        ros_enabled=enable_ros,
    )

    # only now can we import the rest of the modules
    from core.universe import Universe

    # to circumvent Python typing restrictions, we type the universe object here
    universe: Universe = universe

    from core.archiver import Archiver

    # creates a singleton instance of the archiver, to be used throughout the program
    Archiver.create(universe, db_config, secrets["db"])

    from core.envs import NewtonTerrainEnv
    from core.agents import NewtonVecAgent

    from core.sensors import VecIMU, VecContact
    from core.actuators import BaseActuator, DCActuator
    from core.controllers import VecJointsController, CommandController
    from core.animation import AnimationEngine
    from core.domain_randomizer import NewtonBaseDomainRandomizer
    from core.terrain.terrain import Terrain, TerrainType, SubTerrainType

    from core.utils.math import IDENTITY_QUAT

    imu = VecIMU(
        universe=universe,
        local_position=torch.zeros((num_envs, 3)),
        local_orientation=IDENTITY_QUAT.repeat(num_envs, 1),
        noise_function=lambda x: x,
    )

    contact_sensor = VecContact(
        universe=universe,
        num_contact_sensors_per_agent=4,
    )

    actuators: List[BaseActuator] = []

    for i in range(12):
        actuator = DCActuator(
            universe=universe,
            k_p=1.0,
            k_d=0.001,
            effort_saturation=120.0,
        )

        actuators.append(actuator)

    joints_controller = VecJointsController(
        universe=universe,
        joint_position_limits=robot_config["joints"]["limits"]["positions"],
        joint_velocity_limits=robot_config["joints"]["limits"]["velocities"],
        joint_effort_limits=robot_config["joints"]["limits"]["efforts"],
        joint_gear_ratios=robot_config["joints"]["gear_ratios"],
        noise_function=lambda x: x,
        actuators=actuators,
        fixed_joints=robot_config["joints"]["fixed"],
    )

    command_controller = CommandController(universe=universe)

    if enable_ros:
        from core.sensors import ROSVecIMU, ROSVecContact
        from core.controllers import ROSVecJointsController, ROSCommandController
        from core.utils.ros import get_qos_profile_from_node_config

        namespace: str = ros_config["namespace"]

        imu_node_config: Config = ros_config["nodes"]["imu"]
        imu = ROSVecIMU(
            vec_imu=imu,
            node_name=imu_node_config["name"],
            namespace=namespace,
            pub_sim_topic=imu_node_config["pub_sim_topic"],
            pub_real_topic=imu_node_config["pub_real_topic"],
            pub_period=imu_node_config["pub_period"],
            pub_qos_profile=get_qos_profile_from_node_config(
                imu_node_config,
                "pub_qos",
                ros_config,
            ),
        )

        contact_node_config: Config = ros_config["nodes"]["contact"]
        contact_sensor = ROSVecContact(
            vec_contact=contact_sensor,
            node_name=contact_node_config["name"],
            namespace=namespace,
            pub_sim_topic=contact_node_config["pub_sim_topic"],
            pub_real_topic=contact_node_config["pub_real_topic"],
            pub_period=contact_node_config["pub_period"],
            pub_qos_profile=get_qos_profile_from_node_config(
                contact_node_config,
                "pub_qos",
                ros_config,
            ),
        )

        joints_controller_node_config: Config = ros_config["nodes"]["joints_controller"]
        joints_controller = ROSVecJointsController(
            vec_joints_controller=joints_controller,
            node_name=joints_controller_node_config["name"],
            namespace=namespace,
            pub_sim_topic=joints_controller_node_config["pub_sim_topic"],
            pub_real_topic=joints_controller_node_config["pub_real_topic"],
            pub_period=joints_controller_node_config["pub_period"],
            pub_qos_profile=get_qos_profile_from_node_config(
                joints_controller_node_config,
                "pub_qos",
                ros_config,
            ),
        )

        command_controller_node_config: Config = ros_config["nodes"][
            "command_controller"
        ]
        command_controller = ROSCommandController(
            command_controller=command_controller,
            node_name=command_controller_node_config["name"],
            namespace=namespace,
            pub_sim_topic=command_controller_node_config["pub_sim_topic"],
            pub_real_topic=command_controller_node_config["pub_real_topic"],
            pub_period=command_controller_node_config["pub_period"],
            pub_qos_profile=get_qos_profile_from_node_config(
                command_controller_node_config,
                "pub_qos",
                ros_config,
            ),
        )

    newton_agent = NewtonVecAgent(
        universe=universe,
        num_agents=num_envs,
        imu=imu,
        joints_controller=joints_controller,
        contact_sensor=contact_sensor,
    )

    animation_engine = AnimationEngine(
        universe=universe,
        current_clip_config=current_animation_clip_config,
    )

    # TODO: Make the domain randomizer optional
    domain_randomizer = NewtonBaseDomainRandomizer(
        universe=universe,
        seed=158124,
        agent=newton_agent,
        randomizer_settings=randomization_config,
    )

    terrain = Terrain(universe, terrain_config, num_envs)

    # ----------- #
    #    ONNX     #
    # ----------- #

    if exporting:
        raise NotImplementedError("ONNX export is not yet implemented.")

    # --------------- #
    #    ANIMATING    #
    # --------------- #

    if animating:
        env = NewtonTerrainEnv(
            universe=universe,
            agent=newton_agent,
            num_envs=num_envs,
            terrain=terrain,
            domain_randomizer=domain_randomizer,
            inverse_control_frequency=inverse_control_frequency,
        )

        terrain.register_self(
            TerrainType.Specific,
            1,  # num_rows
            1,  # num_cols
            SubTerrainType.RandomUniform,
        )  # done manually, since we're changing some default construction parameters
        env.register_self()  # done manually, generally the task would do this
        animation_engine.register_self()  # done manually, generally the task would do this

        universe.construct_registrations()

        env.reset()  # reset the environment to get correctly position the agent

        ordered_dof_names = joints_controller.art_view.dof_names

        # this is very specific to Newton, because we know that it takes joint positions and the animation engine
        # provides that exactly; a different robot or different control mode would probably require a different approach
        while universe.is_playing:
            joint_data = animation_engine.get_multiple_clip_data_at_seconds(
                torch.tensor([universe.current_time]),
                ordered_dof_names,
            )

            # index 7 is the joint position (angle in degrees)
            joint_positions = joint_data[0, :, 7]

            # TODO: Investigate if there's a way to simplify joint_normalization
            #   since the system expects a [-1, 1] range, we normalize the joint positions to their limits; it is
            #   redundant since we'll undo the normalization later in the controller, so it might warrant a change
            joint_actions = newton_agent.joints_controller.normalize_joint_positions(
                joint_positions
            )

            # we need to make it 2D, since the controller expects a batch of actions
            env.step(joint_actions.unsqueeze(0))

        return

    # ---------------- #
    #   PHYSICS ONLY   #
    # ---------------- #

    if physics_only:
        env = NewtonTerrainEnv(
            universe=universe,
            agent=newton_agent,
            num_envs=num_envs,
            terrain=terrain,
            domain_randomizer=domain_randomizer,
            inverse_control_frequency=inverse_control_frequency,
        )

        terrain.register_self(
            TerrainType.Random,
            1,  # num_rows
            1,  # num_cols
        )  # done manually, since we're changing some default construction parameters
        env.register_self()  # done manually, generally the task would do this

        universe.construct_registrations()

        env.reset()  # reset the environment to get correctly position the agent

        while universe.is_playing:
            env.step(torch.zeros((num_envs, 12)))

        exit(1)

    # ----------- #
    #     RL      #
    # ----------- #

    from core.tasks import (
        NewtonBaseTask,
        NewtonIdleTask,
        NewtonLocomotionTask,
    )

    training_env = NewtonTerrainEnv(
        universe=universe,
        agent=newton_agent,
        num_envs=num_envs,
        terrain=terrain,
        domain_randomizer=domain_randomizer,
        inverse_control_frequency=inverse_control_frequency,
    )

    playing_env = NewtonTerrainEnv(
        universe=universe,
        agent=newton_agent,
        num_envs=num_envs,
        terrain=terrain,
        domain_randomizer=domain_randomizer,
        inverse_control_frequency=inverse_control_frequency,
    )

    # task used for either training or playing
    task: NewtonBaseTask

    if current_task_name == "newton_idle":
        task = NewtonIdleTask(
            universe=universe,
            env=playing_env if playing else training_env,
            agent=newton_agent,
            animation_engine=animation_engine,
            device=current_task_config["device"],
            num_envs=num_envs,
            playing=playing,
            reset_in_play=current_task_config["reset_in_play"],
            max_episode_length=current_task_config["episode_length"],
            observation_scalers=current_task_config["scalers"]["observations"],
            action_scaler=current_task_config["scalers"]["action"],
            reward_scalers=current_task_config["scalers"]["rewards"],
        )
    elif current_task_name == "newton_locomotion":
        task = NewtonLocomotionTask(
            universe=universe,
            env=playing_env if playing else training_env,
            agent=newton_agent,
            animation_engine=animation_engine,
            command_controller=command_controller,
            device=current_task_config["device"],
            num_envs=num_envs,
            playing=playing,
            reset_in_play=current_task_config["reset_in_play"],
            max_episode_length=current_task_config["episode_length"],
            observation_scalers=current_task_config["scalers"]["observations"],
            action_scaler=current_task_config["scalers"]["action"],
            reward_scalers=current_task_config["scalers"]["rewards"],
            command_scalers=current_task_config["scalers"]["commands"],
        )
    else:
        Logger.error(f"Task {current_task_name} not recognized.")
        return

    if training or playing:
        from skrl.utils import set_seed

        set_seed()

        from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
        from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG

        from core.utils.rl.config import parse_ppo_config

        ppo_config = parse_ppo_config(current_task_config["ppo"], PPO_DEFAULT_CONFIG)
        ppo_config["state_preprocessor_kwargs"] = {
            "size": task.observation_space,
            "device": task.device,
        }
        ppo_config["value_preprocessor_kwargs"] = {
            "size": 1,
            "device": task.device,
        }
        ppo_config["experiment"]["directory"] = runs_dir
        ppo_config["experiment"]["experiment_name"] = current_run_name

        trainer_config = SEQUENTIAL_TRAINER_DEFAULT_CONFIG.copy()
        trainer_config["timesteps"] = current_task_config["timesteps"]
        trainer_config["headless"] = headless
        trainer_config["stochastic_evaluation"] = False  # deterministic evaluation

        from core.utils.rl.skrl import (
            create_shared_model,
            create_ppo,
            create_random_memory,
            create_sequential_trainer,
        )

        model = create_shared_model(
            task=task,
            arch=network_config["net_arch"],
            activation=network_config["activation_fn"],
        )

        random_memory = create_random_memory(
            task=task,
            memory_size=current_task_config["ppo"]["n_steps"],
        )

        algo = create_ppo(
            task=task,
            ppo_config=ppo_config,
            memory=random_memory,
            policy_model=model,
            checkpoint_path=current_checkpoint_path,
        )

        trainer = create_sequential_trainer(
            task=task,
            algorithm=algo,
            trainer_config=trainer_config,
        )

        universe.construct_registrations()

        from core.utils.config import record_configs

        record_directory = os.path.join(runs_dir, current_run_name, "records")
        configs_to_record = {
            "task_config": current_task_config,
            "world_config": world_config,
            "robot_config": robot_config,
            "network_config": network_config,
            "randomization_config": randomization_config,
            "animation_clip_config": current_animation_clip_config,
        }
        record_configs(record_directory, configs_to_record)

        if training:
            trainer.train()
            return

        trainer.eval()

        return


if __name__ == "__main__":
    main()
