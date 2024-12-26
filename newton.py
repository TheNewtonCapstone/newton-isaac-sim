import argparse
from typing import List, Optional, Tuple

from core.types import Matter, Settings, SettingsCollection


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
        "--world-config",
        type=str,
        help="Path to the configuration file for the world.",
        default="configs/world.yaml",
    )
    parser.add_argument(
        "--randomization-config",
        type=str,
        help="Enable domain randomization.",
        default="configs/randomization.yaml",
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
        "--disable-ros2",
        action="store_true",
        help="Enable ROS2 support.",
        default=False,
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
) -> Optional[Tuple[SettingsCollection, str]]:
    from core.utils.path import get_files_with_extension
    from core.utils.config import animation_configs_to_clips_settings

    # discover animations in directory
    animation_configs_filenames = get_files_with_extension(
        animation_config_dir, ".yaml"
    )
    animation_configs_paths = [
        f"{animation_config_dir}/{f}" for f in animation_configs_filenames
    ]

    if len(animation_configs_paths) == 0:
        print(f"No animation files found in {animation_config_dir}.")
        return None

    # load animation clips settings from config files
    animation_clips_settings = animation_configs_to_clips_settings(
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
    current_checkpoint_name: Optional[str],
    required: bool,
) -> Optional[Tuple[Settings, str]]:
    from core.utils.checkpoints import (
        save_runs_library,
        build_runs_library_from_runs_folder,
    )

    runs_settings: Settings = build_runs_library_from_runs_folder(runs_dir)

    save_runs_library(runs_settings, runs_dir)

    if current_checkpoint_name and current_checkpoint_name in runs_settings:
        return (
            runs_settings,
            runs_settings[current_checkpoint_name]["saves"][-1]["path"],
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
        prompt="Please enter a run name: ",
        default=list(runs_settings.keys())[-1],
        strip=True,
    )

    selected_checkpoint_name = ""

    while selected_checkpoint_name not in runs_settings:
        selected_checkpoint_name = cli.launch()

    return (
        runs_settings,
        runs_settings[selected_checkpoint_name]["saves"][-1]["path"],
    )


def setup() -> Optional[Matter]:
    from core.utils.config import load_config

    parser = setup_argparser()

    # General Configs
    cli_args, _ = parser.parse_known_args()

    # Control Flags
    training = cli_args.train
    playing = cli_args.play
    animating = cli_args.animate
    physics_only = cli_args.physics
    exporting = cli_args.export_onnx

    _modes = {
        "Training": training,
        "Playing": playing,
        "Animating": animating,
        "Physics Only": physics_only,
        "Exporting": exporting,
    }

    none_selected = sum(_modes.values()) == 0
    multiple_selected = sum(_modes.values()) > 1

    if multiple_selected:
        print(
            "Only one of training, playing, animating, physics or exporting can be set."
        )
        return None
    elif none_selected:
        mode_name, mode_idx = mode_select(list(_modes.keys()))

        training = mode_idx == 0
        playing = mode_idx == 1
        animating = mode_idx == 2
        physics_only = mode_idx == 3
        exporting = mode_idx == 4
    else:
        _modes_list = list(_modes.values())
        _modes_flag_idx = _modes_list.index(True)

        mode_name = list(_modes.keys())[_modes_flag_idx]

    # Animation

    animation_clips_config: Settings = {}
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

    # Checkpoint

    runs_library: Settings = {}
    runs_dir: str = cli_args.runs_dir
    current_checkpoint_path: Optional[str] = cli_args.checkpoint_path
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
            cli_args.checkpoint_name,
            required=not training,
        )

        if checkpoint_select_result is None and (playing or exporting):
            return None

        if checkpoint_select_result is not None:
            runs_library, current_checkpoint_path = checkpoint_select_result

    # Configs

    robot_config = load_config(cli_args.robot_config)
    rl_config = load_config(cli_args.rl_config)
    world_config = load_config(cli_args.world_config)
    randomization_config = load_config(cli_args.randomization_config)

    # Helper flags

    is_rl = training or playing
    headless = (
        cli_args.headless or cli_args.export_onnx
    )  # if we're exporting, don't show the GUI
    interactive = not headless and (
        physics_only or playing or animating
    )  # interactive means that the user is expected to control the agent in some way
    enable_ros2 = cli_args.disable_ros2

    # override some config with CLI num_envs, if specified
    num_envs = rl_config["n_envs"]
    if cli_args.num_envs != -1:
        num_envs = cli_args.num_envs
    if not training:
        num_envs = 1

    if interactive:
        world_config["sim_params"]["enable_scene_query_support"] = True

    control_step_dt = (
        rl_config["newton"]["inverse_control_frequency"] * world_config["physics_dt"]
    )

    return (
        cli_args,
        robot_config,
        rl_config,
        world_config,
        randomization_config,
        animation_clips_config,
        current_animation,
        runs_dir,
        runs_library,
        current_checkpoint_path,
        mode_name,
        training,
        playing,
        animating,
        physics_only,
        exporting,
        is_rl,
        interactive,
        headless,
        enable_ros2,
        num_envs,
        control_step_dt,
    )


def main():
    base_matter = setup()

    if base_matter is None:
        print("An error occurred during setup.")
        return

    (
        cli_args,
        robot_config,
        rl_config,
        world_config,
        randomization_config,
        animation_clips_config,
        current_animation,
        runs_dir,
        runs_library,
        current_checkpoint_path,
        mode_name,
        training,
        playing,
        animating,
        physics_only,
        exporting,
        is_rl,
        interactive,
        headless,
        enable_ros2,
        num_envs,
        control_step_dt,
    ) = base_matter

    print(
        f"Running with {num_envs} environments, {rl_config['ppo']['n_steps']} steps per environment, and {'headless' if headless else 'GUI'} mode.\n",
        f"{mode_name}{(' (with checkpoint ' + current_checkpoint_path + ')') if current_checkpoint_path is not None else ''}.\n",
        f"Using {rl_config['device']} as the RL device and {world_config['device']} as the physics device.",
    )

    import torch
    import numpy as np

    # big_bang must be imported & invoked first, to load all necessary omniverse extensions
    from core import big_bang

    universe = big_bang({"headless": headless}, world_config, disable_ros2=enable_ros2)

    # only now can we import the rest of the modules
    from core.universe import Universe

    # to circumvent Python typing restrictions, we type the universe here
    universe: Universe = universe

    from core.envs import NewtonMultiTerrainEnv
    from core.agents import NewtonVecAgent

    from core.terrain.flat_terrain import FlatBaseTerrainBuilder
    from core.terrain.perlin_terrain import PerlinBaseTerrainBuilder

    from core.sensors import VecIMU
    from core.sensors.contact import VecContact
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

    joints_controller = VecJointsController(
        universe=universe,
        joint_position_limits=robot_config["joints"]["limits"]["positions"],
        joint_velocity_limits=robot_config["joints"]["limits"]["velocities"],
        noise_function=lambda x: x,
    )

    contact_sensor = VecContact(
        universe=universe,
        num_contact_sensors_per_agent=4,
    )

    newton_agent = NewtonVecAgent(
        num_agents=num_envs,
        imu=imu,
        joints_controller=joints_controller,
        contact_sensor=contact_sensor,
    )

    animation_engine = AnimationEngine(
        clips=animation_clips_config,
    )

    domain_randomizer = NewtonBaseDomainRandomizer(
        seed=rl_config["seed"],
        agent=newton_agent,
        randomizer_settings=randomization_config,
    )

    # --------------- #
    #    ANIMATING    #
    # --------------- #

    if animating:
        env = NewtonMultiTerrainEnv(
            agent=newton_agent,
            num_envs=num_envs,
            terrain_builders=[
                FlatBaseTerrainBuilder(),
                FlatBaseTerrainBuilder(),
                FlatBaseTerrainBuilder(),
                FlatBaseTerrainBuilder(),
            ],
            domain_randomizer=domain_randomizer,
            inverse_control_frequency=rl_config["newton"]["inverse_control_frequency"],
        )

        env.construct(universe)
        env.reset()  # called manually, because the task usually does it, must be done before stepping

        animation_engine.construct(current_animation)

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

            # we wrap it in an array to make it 2D (it's a vectorized env)
            env.step(joint_actions.unsqueeze(0))

        exit(1)

    # ---------------- #
    #   PHYSICS ONLY   #
    # ---------------- #

    if physics_only:
        env = NewtonMultiTerrainEnv(
            agent=newton_agent,
            num_envs=num_envs,
            terrain_builders=[PerlinBaseTerrainBuilder(), FlatBaseTerrainBuilder()],
            domain_randomizer=domain_randomizer,
            inverse_control_frequency=rl_config["newton"]["inverse_control_frequency"],
        )

        env.construct(universe)
        env.reset()

        while universe.is_playing:
            env.step(np.zeros((num_envs, 12)))

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
        inverse_control_frequency=rl_config["newton"]["inverse_control_frequency"],
    )

    playing_env = NewtonMultiTerrainEnv(
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
        inverse_control_frequency=rl_config["newton"]["inverse_control_frequency"],
    )

    # TODO: Improve the way we save runs
    #   Maybe have a database? Or a file that keeps track of all runs with some metadata, like the date, the agent used,
    #   the task, the environment, etc. We would also have a way to easily load the last run, or a specific run.
    #   labels: enhancement

    from core.utils.checkpoints import (
        get_unused_run_id,
        create_runs_library,
    )

    new_checkpoint_id = get_unused_run_id(runs_library)

    if new_checkpoint_id is None:
        runs_library = create_runs_library(runs_dir)

        new_checkpoint_id = get_unused_run_id(runs_library)

    run_name = f"newton_idle_{new_checkpoint_id}"

    # task used for either training or playing
    task = NewtonIdleTask(
        training_env=training_env,
        playing_env=playing_env,
        agent=newton_agent,
        animation_engine=animation_engine,
        device=rl_config["device"],
        num_envs=num_envs,
        playing=playing,
        max_episode_length=rl_config["episode_length"],
        rl_step_dt=control_step_dt,
    )
    callback = NewtonBaseTaskCallback(
        check_freq=64,
        save_path=run_name,
    )

    task.construct(universe)

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

        # TODO: Add proper way to customize network size & shape
        #   We can offer a set of predefined policies, or allow the user to specify their own. I'm thinking of a
        #   set of functions, instead of classes, since it's a small configuration. We could also offer a wrapper around
        #   PPO (and eventually A2C) to allow for easy customization of the network.

        policy_kwargs = dict(activation_fn=torch.nn.ELU, net_arch=[1024, 1024, 1024])

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
            tb_log_name=run_name,
            reset_num_timesteps=False,
            progress_bar=True,
            callback=callback,
        )
        model.save(f"{runs_dir}/{run_name}_0/model.zip")

        exit(1)

    if playing:
        from core.utils.path import get_folder_from_path

        model = PPO.load(current_checkpoint_path)

        actions = model.predict(task.reset()[0], deterministic=True)[0]
        actions = np.array([actions])  # make sure we have a 2D tensor

        log_file = open(
            f"{get_folder_from_path(current_checkpoint_path)}/playing.csv", "w"
        )
        print("time,dt,roll,action1,action2", file=log_file)

        while universe.is_playing:
            step_return = task.step(actions)
            observations = step_return[0]

            actions = model.predict(observations, deterministic=True)[0]
            actions_string = ",".join([str(ja) for ja in actions[0]])

            print(
                f"{universe.current_time},{universe.get_physics_dt()},{observations[0][0]},{actions_string}",
                file=log_file,
            )

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

    print(f"Exported to {current_checkpoint_path}.onnx!")


if __name__ == "__main__":
    main()
