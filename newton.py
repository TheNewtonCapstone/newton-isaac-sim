import argparse
import logging
from typing import List, Optional, Tuple, get_args

from core.types import Matter, Config, ConfigCollection, Mode

logger = logging.getLogger(__name__)
logging.basicConfig(filename="newton.log", level=logging.INFO)


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
        "--rl-config",
        type=str,
        help="Path to the configuration file for RL.",
        default="configs/tasks/newton_idle_task.yaml",
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
        "--secrets",
        type=str,
        help="Path to the secrets file (containing database token, API keys, etc.).",
        default="configs/secrets.yaml",
    )
    parser.add_argument(
        "--animations-dir",
        type=str,
        help="Path to the directory containing animations.",
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


# TODO: Implement selection of tasks (e.g. NewtonIdleTask, NewtonWalkTask, etc.)
#   We can read the tasks from a directory and offer that as a selection to the user.
def task_select():
    pass


def animation_select(
    animation_config_dir: str,
    current_animation_name: Optional[str],
) -> Optional[Tuple[ConfigCollection, str]]:
    from core.utils.path import get_files_with_extension
    from core.utils.config import animation_configs_to_clips_config

    # discover animations in directory
    animation_configs_filenames = get_files_with_extension(
        animation_config_dir, ".yaml"
    )
    animation_configs_paths = [
        f"{animation_config_dir}/{f}" for f in animation_configs_filenames
    ]

    if len(animation_configs_paths) == 0:
        logger.info(f"No animation files found in {animation_config_dir}.")
        return None

    # load animation clips settings from config files
    animation_clips_settings = animation_configs_to_clips_config(
        animation_configs_paths
    )

    # discard any duplicate animation clips by name, taking the last one
    animation_clips_settings = {
        clip["name"]: clip for clip in animation_clips_settings.values()
    }
    animation_clips_names = [clip["name"] for clip in animation_clips_settings.values()]

    # use the specified animation if it exists
    if current_animation_name and current_animation_name in animation_clips_settings:
        return animation_clips_settings, current_animation_name

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
        animation_clips_settings,
        list(animation_clips_settings.values())[selected_animation_idx]["name"],
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
        logger.info(f"No runs found in {runs_dir}.")
        return None

    save_runs_library(runs_settings, runs_dir)

    if current_run_name and current_run_name in runs_settings:
        return (
            runs_settings,
            current_run_name,
            runs_settings[current_run_name]["checkpoints"][-1]["path"],
        )

    from bullet import Input, YesNo

    if not required:
        cli = YesNo(
            prompt="Would you like to select a checkpoint? ",
            default="n",
        )

        if not cli.launch():
            return None

    most_recent_checkpoint = str(list(runs_settings.keys())[-1])

    cli = Input(
        prompt=f"Please enter a run name (format: <task_name>_001): ",
        default=f"",
        strip=True,
    )

    selected_run_name = ""

    while selected_run_name not in runs_settings:
        selected_run_name = cli.launch()

        if len(selected_run_name) == 0:
            print(f"Selecting latest checkpoint: {most_recent_checkpoint}.")
            selected_run_name = most_recent_checkpoint

    return (
        runs_settings,
        selected_run_name,
        runs_settings[selected_run_name]["checkpoints"][-1]["path"],
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


def setup() -> Optional[Matter]:
    from core.utils.config import load_config

    parser = setup_argparser()

    # General Configs
    cli_args, _ = parser.parse_known_args()

    robot_config = load_config(cli_args.robot_config)
    rl_config = load_config(cli_args.rl_config)
    world_config = load_config(cli_args.world_config)
    randomization_config = load_config(cli_args.randomization_config)
    ros_config = load_config(cli_args.ros_config)
    db_config = load_config(cli_args.db_config)
    secrets = load_config(cli_args.secrets)

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
        logger.info(
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

    # Animation

    animation_clips_config: Config = {}
    current_animation: Optional[str] = None

    if animating or training or playing:
        animation_select_result = animation_select(
            cli_args.animations_dir,
            cli_args.animation_name,
        )

        # We failed to find an animation, exit
        if animation_select_result is None:
            return None

        animation_clips_config, current_animation = animation_select_result

    # Run & checkpoint config

    runs_library: Config = {}
    runs_dir: str = cli_args.runs_dir

    current_checkpoint_path: Optional[str] = cli_args.checkpoint_path
    current_run_name: Optional[str] = cli_args.run_name
    no_checkpoint = cli_args.no_checkpoint

    if (playing or exporting) and no_checkpoint:
        print("No checkpoint specified for playing or exporting.")
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

        if checkpoint_select_result is None and (playing or exporting):
            return None

        if checkpoint_select_result is not None:
            runs_library, current_run_name, current_checkpoint_path = (
                checkpoint_select_result
            )

    # if we're training and the user did specify no-checkpoint, we need to create a new run
    if no_checkpoint and training and current_run_name is None:
        from core.utils.runs import (
            get_unused_run_id,
            create_runs_library,
        )

        new_run_id = get_unused_run_id(runs_library)

        if new_run_id is None:
            runs_library = create_runs_library(runs_dir)

            new_run_id = get_unused_run_id(runs_library)

        current_run_name = f"{rl_config['task_name']}_{new_run_id}"

    # Network config

    network_config: Config = {}
    network_name: Optional[str] = cli_args.network_name

    if training:
        network_select_result = network_arch_select(
            cli_args.network_config, network_name
        )

        if network_select_result is None:
            print("No network architecture found.")
            return None

        network_config, network_name = network_select_result

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

    # override some config with CLI num_envs, if specified
    num_envs = rl_config["n_envs"]
    if cli_args.num_envs != -1:
        num_envs = cli_args.num_envs
    if not training:
        num_envs = 1

    if interactive:
        world_config["sim_params"]["enable_scene_query_support"] = True

    control_step_dt = world_config["control_dt"]
    inverse_control_frequency = int(control_step_dt / world_config["physics_dt"])

    return (
        cli_args,
        robot_config,
        rl_config,
        world_config,
        randomization_config,
        network_config,
        network_name,
        ros_config,
        db_config,
        secrets,
        animation_clips_config,
        current_animation,
        runs_dir,
        runs_library,
        current_run_name,
        current_checkpoint_path,
        mode,
        mode_name,
        training,
        playing,
        animating,
        physics_only,
        exporting,
        is_rl,
        interactive,
        headless,
        enable_ros,
        enable_db,
        num_envs,
        control_step_dt,
        inverse_control_frequency,
    )


def main():
    base_matter = setup()

    if base_matter is None:
        logger.info("An error occurred during setup.")
        return

    (
        cli_args,
        robot_config,
        rl_config,
        world_config,
        randomization_config,
        network_config,
        network_name,
        ros_config,
        db_config,
        secrets,
        animation_clips_config,
        current_animation,
        runs_dir,
        runs_library,
        current_run_name,
        current_checkpoint_path,
        mode,
        mode_name,
        training,
        playing,
        animating,
        physics_only,
        exporting,
        is_rl,
        interactive,
        headless,
        enable_ros,
        enable_db,
        num_envs,
        control_step_dt,
        inverse_control_frequency,
    ) = base_matter

    logger.info(
        f"Running with {num_envs} environments, {rl_config['ppo']['n_steps']} steps per environment, and {'headless' if headless else 'GUI'} mode.\n",
        f"{mode_name}{(' (with checkpoint ' + current_checkpoint_path + ')') if current_checkpoint_path is not None else ''}.\n",
        f"Using {rl_config['device']} as the RL device and {world_config['device']} as the physics device.",
    )

    import torch
    import numpy as np

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

    from core.envs import NewtonMultiTerrainEnv
    from core.agents import NewtonVecAgent

    from core.terrain.flat_terrain import FlatBaseTerrainBuilder
    from core.terrain.perlin_terrain import PerlinBaseTerrainBuilder

    from core.sensors import VecIMU, VecContact
    from core.actuators import DCActuator
    from core.controllers import VecJointsController
    from core.animation import AnimationEngine
    from core.domain_randomizer import NewtonBaseDomainRandomizer

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

    actuators: List[DCActuator] = []
    effort_saturation_config: Config = robot_config["actuators"]["dc"][
        "effort_saturation"
    ]
    gain_config_list: List[Config] = list(
        robot_config["actuators"]["dc"]["gains"].values()
    )

    for i in range(12):
        actuator = DCActuator(
            universe=universe,
            k_p=gain_config_list[i]["p"],
            k_d=gain_config_list[i]["d"],
            effort_saturation=list(effort_saturation_config.values())[i],
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

    if enable_ros:
        from core.sensors import ROSVecIMU, ROSVecContact
        from core.controllers import ROSVecJointsController
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

    newton_agent = NewtonVecAgent(
        universe=universe,
        num_agents=num_envs,
        imu=imu,
        joints_controller=joints_controller,
        contact_sensor=contact_sensor,
    )

    animation_engine = AnimationEngine(
        universe=universe,
        clips=animation_clips_config,
    )

    domain_randomizer = NewtonBaseDomainRandomizer(
        universe=universe,
        seed=rl_config["seed"],
        agent=newton_agent,
        randomizer_settings=randomization_config,
    )

    # --------------- #
    #    ANIMATING    #
    # --------------- #

    if animating:
        env = NewtonMultiTerrainEnv(
            universe=universe,
            agent=newton_agent,
            num_envs=num_envs,
            terrain_builders=[
                FlatBaseTerrainBuilder(),
                FlatBaseTerrainBuilder(),
                FlatBaseTerrainBuilder(),
                FlatBaseTerrainBuilder(),
            ],
            domain_randomizer=domain_randomizer,
            inverse_control_frequency=inverse_control_frequency,
        )

        env.register_self()  # done manually, generally the task would do this
        animation_engine.register_self(current_animation)

        universe.reset(construction=True)

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
            env.step(joint_actions.repeat(1, 1))

        exit(1)

    # ---------------- #
    #   PHYSICS ONLY   #
    # ---------------- #

    if physics_only:
        env = NewtonMultiTerrainEnv(
            universe=universe,
            agent=newton_agent,
            num_envs=num_envs,
            terrain_builders=[
                FlatBaseTerrainBuilder(),
                FlatBaseTerrainBuilder(),
                FlatBaseTerrainBuilder(),
                FlatBaseTerrainBuilder(),
            ],
            domain_randomizer=domain_randomizer,
            inverse_control_frequency=inverse_control_frequency,
        )

        env.register_self()  # done manually, generally the task would do this

        universe.reset(construction=True)

        while universe.is_playing:
            env.step(torch.zeros((num_envs, 12)))

        exit(1)

    # ----------- #
    #     RL      #
    # ----------- #

    from core.tasks import NewtonIdleTask, NewtonBaseTaskCallback
    from core.envs import NewtonMultiTerrainEnv
    from core.wrappers import RandomDelayWrapper

    terrains_size = torch.tensor([10, 10])
    terrains_resolution = torch.tensor([20, 20])

    training_env = NewtonMultiTerrainEnv(
        universe=universe,
        agent=newton_agent,
        num_envs=num_envs,
        terrain_builders=[
            FlatBaseTerrainBuilder(size=terrains_size),
            PerlinBaseTerrainBuilder(
                size=terrains_size,
                resolution=terrains_resolution,
                height=0.05,
                octave=4,
                noise_scale=2,
            ),
            PerlinBaseTerrainBuilder(
                size=terrains_size,
                resolution=terrains_resolution,
                height=0.03,
                octave=8,
                noise_scale=4,
            ),
            PerlinBaseTerrainBuilder(
                size=terrains_size,
                resolution=terrains_resolution,
                height=0.02,
                octave=16,
                noise_scale=8,
            ),
        ],
        domain_randomizer=domain_randomizer,
        inverse_control_frequency=inverse_control_frequency,
    )

    playing_env = NewtonMultiTerrainEnv(
        universe=universe,
        agent=newton_agent,
        num_envs=num_envs,
        terrain_builders=[
            FlatBaseTerrainBuilder(size=terrains_size),
            PerlinBaseTerrainBuilder(
                size=terrains_size,
                resolution=terrains_resolution,
                height=0.05,
                octave=4,
                noise_scale=2,
            ),
            PerlinBaseTerrainBuilder(
                size=terrains_size,
                resolution=terrains_resolution,
                height=0.03,
                octave=8,
                noise_scale=4,
            ),
            PerlinBaseTerrainBuilder(
                size=terrains_size,
                resolution=terrains_resolution,
                height=0.02,
                octave=16,
                noise_scale=8,
            ),
        ],
        domain_randomizer=domain_randomizer,
        inverse_control_frequency=inverse_control_frequency,
    )

    # task used for either training or playing
    task = NewtonIdleTask(
        universe=universe,
        env=playing_env if playing else training_env,
        agent=newton_agent,
        animation_engine=animation_engine,
        device=rl_config["device"],
        num_envs=num_envs,
        playing=playing,
        max_episode_length=rl_config["episode_length"],
    )
    callback = NewtonBaseTaskCallback(
        check_freq=64,
        save_path=current_run_name,
    )

    universe.reset(construction=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import BasePolicy

    # we're not exporting nor purely simulating, so we're training
    if training:
        if rl_config["delay"]["enabled"]:
            list_obs_delay_range = rl_config["delay"]["obs_delay_range"]
            list_act_delay_range = rl_config["delay"]["act_delay_range"]
            instant_rewards = rl_config["delay"]["instant_rewards"]

            obs_delay_range = range(list_obs_delay_range[0], list_obs_delay_range[1])
            act_delay_range = range(list_act_delay_range[0], list_act_delay_range[1])

            task = RandomDelayWrapper(
                task,
                obs_delay_range=obs_delay_range,
                act_delay_range=act_delay_range,
                instant_rewards=instant_rewards,
            )

        policy_kwargs = dict(
            activation_fn=network_config["activation_fn"],
            net_arch=network_config["net_arch"],
        )

        model = PPO(
            rl_config["policy"],
            task,
            verbose=2,
            device=rl_config["device"],
            seed=rl_config["seed"],
            learning_rate=float(rl_config["base_lr"]),
            n_steps=rl_config["ppo"]["n_steps"],
            batch_size=rl_config["ppo"]["batch_size"],
            n_epochs=rl_config["ppo"]["n_epochs"],
            gamma=rl_config["ppo"]["gamma"],
            gae_lambda=rl_config["ppo"]["gae_lambda"],
            clip_range=float(rl_config["ppo"]["clip_range"]),
            clip_range_vf=rl_config["ppo"]["clip_range_vf"],
            ent_coef=rl_config["ppo"]["ent_coef"],
            vf_coef=rl_config["ppo"]["vf_coef"],
            max_grad_norm=rl_config["ppo"]["max_grad_norm"],
            use_sde=rl_config["ppo"]["use_sde"],
            sde_sample_freq=rl_config["ppo"]["sde_sample_freq"],
            target_kl=rl_config["ppo"]["target_kl"],
            tensorboard_log=runs_dir,
            policy_kwargs=policy_kwargs,
        )

        if current_checkpoint_path is not None:
            model = PPO.load(current_checkpoint_path, task, device=rl_config["device"])

        model.learn(
            total_timesteps=rl_config["timesteps_per_env"] * num_envs,
            tb_log_name=current_run_name,
            reset_num_timesteps=False,
            progress_bar=True,
            callback=callback,
        )
        model.save(f"{runs_dir}/{current_run_name}_0/model.zip")

        exit(1)

    if playing:
        from core.utils.path import get_folder_from_path

        model = PPO.load(current_checkpoint_path)

        actions = model.predict(task.reset()[0], deterministic=True)[0]
        actions = np.array([actions])  # make sure we have a 2D tensor

        log_file = open(
            f"{get_folder_from_path(current_checkpoint_path)}/playing.csv", "w"
        )
        # logger.info("time,dt,roll,action1,action2", file=log_file)

        while universe.is_playing:
            step_return = task.step(actions)
            observations = step_return[0]

            actions = model.predict(observations, deterministic=True)[0]
            actions_string = ",".join([str(ja) for ja in actions[0]])

            # logger.info(
            #     f"{universe.current_time},{universe.get_physics_dt()},{observations[0][0]},{actions_string}",
            #     file=log_file,
            # )

        exit(1)

    # ----------- #
    #    ONNX     #
    # ----------- #

    # Load model from checkpoint
    model = PPO.load(current_checkpoint_path, device="cpu")

    # Create dummy observations tensor for tracing torch model
    obs_shape = model.observation_space.shape
    dummy_input = torch.rand((1, *obs_shape), device="cpu")

    # Simplified network for actor inference
    # Tested for continuous_a2c_logstd
    class OnnxablePolicy(torch.nn.Module):
        def __init__(self, policy: BasePolicy):
            super().__init__()
            self.policy = policy

        def forward(
            self,
            observation: torch.Tensor,
        ):
            return self.policy(observation, deterministic=True)

    onnxable_model = OnnxablePolicy(model.policy)
    torch.onnx.export(
        onnxable_model,
        dummy_input,
        f"{current_checkpoint_path}.onnx",
        verbose=True,
        input_names=["observations"],
        output_names=["actions"],
    )  # outputs are mu (actions), sigma, value

    logger.info(f"Exported to {current_checkpoint_path}.onnx!")


if __name__ == "__main__":
    main()
