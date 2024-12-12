import argparse
from typing import List, Optional, Dict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

import torch
from core.types import Matter, Settings
from core.utils.config import load_config
from core.utils.math import IDENTITY_QUAT


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newton.py", description="Entrypoint for any Newton-related actions."
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode.", default=False
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Run the simulation only (no RL).",
        default=False,
    )
    parser.add_argument(
        "--pid-control",
        action="store_true",
        help="Run the simulation with pid control (no RL).",
        default=False,
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
        "--checkpoint",
        type=str,
        help="Path to the checkpoint to load for RL.",
        default=None,
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the agent using a trained checkpoint.",
        default=False,
    )
    parser.add_argument(
        "--animation",
        type=str,
        help="Animate the agent using a parsed animation file.",
        default=None,
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Number of environments to run (will be read from the rl-config if not specified).",
        default=-1,
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Exports checkpoint as ONNX model.",
        default=False,
    )

    return parser


def animation_setting_files_to_clips(
    files: List[str],
) -> Dict[str, Settings]:
    clips = {}

    for file in files:
        clip = load_config(file)
        clips[clip["name"]] = clip

    return clips


def setup() -> Optional[Matter]:
    from core.utils.path import get_files_with_extension

    parser = setup_argparser()

    # General Configs
    cli_args, _ = parser.parse_known_args()
    rl_config = load_config(cli_args.rl_config)
    world_config = load_config(cli_args.world_config)
    randomization_config = load_config(cli_args.randomization_config)

    # Animations
    animation_config_dir = cli_args.animations_dir
    animation_config = [
        f"{animation_config_dir}/{f}"
        for f in get_files_with_extension(animation_config_dir, ".yaml")
    ]

    if len(animation_config) == 0:
        print(f"No animation files found in {animation_config_dir}.")

    animation_clips_config = animation_setting_files_to_clips(animation_config)
    current_animation = cli_args.animation

    # Control Flags
    animating = current_animation is not None
    physics_only = cli_args.physics
    exporting = cli_args.export_onnx
    is_rl = not physics_only and not animating and not exporting
    playing = is_rl and cli_args.play
    training = is_rl and not cli_args.play
    headless = (
        cli_args.headless or cli_args.export_onnx
    )  # if we're exporting, don't show the GUI
    interactive = not headless and (
        physics_only or playing or animating
    )  # interactive means that the user is expected to control the agent in some way

    # override some config with CLI num_envs, if specified
    num_envs = rl_config["n_envs"]
    if cli_args.num_envs != -1:
        num_envs = cli_args.num_envs
    if not training:
        num_envs = 1

    if interactive:
        world_config["sim_params"]["enable_scene_query_support"] = True

    # ensure proper config reading when encountering None
    if rl_config["ppo"]["clip_range_vf"] == "None":
        rl_config["ppo"]["clip_range_vf"] = None

    if rl_config["ppo"]["target_kl"] == "None":
        rl_config["ppo"]["target_kl"] = None

    control_step_dt = (
        rl_config["newton"]["inverse_control_frequency"] * world_config["physics_dt"]
    )

    return (
        cli_args,
        rl_config,
        world_config,
        randomization_config,
        animation_clips_config,
        current_animation,
        animating,
        physics_only,
        is_rl,
        exporting,
        playing,
        training,
        interactive,
        headless,
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
        rl_config,
        world_config,
        randomization_config,
        animation_clips_config,
        current_animation,
        animating,
        physics_only,
        is_rl,
        exporting,
        playing,
        training,
        interactive,
        headless,
        num_envs,
        control_step_dt,
    ) = base_matter

    print(
        f"Running with {num_envs} environments, {rl_config['ppo']['n_steps']} steps per environment, and {'headless' if headless else 'GUI'} mode.\n",
        f"{'Exporting ONNX' if exporting else 'Playing' if playing else 'Training' if training else 'Animating' if animating else 'Physiccing'}.\n",
        f"Using {rl_config['device']} as the RL device and {world_config['device']} as the physics device.",
    )

    # big_bang must be imported & invoked first, to load all necessary omniverse extensions
    from core import big_bang

    universe = big_bang({"headless": headless}, world_config)

    # only now can we import the rest of the modules
    from core.universe import Universe

    # to circumvent Python typing restrictions, we type the universe here
    universe: Universe = universe

    from core.envs import NewtonMultiTerrainEnv
    from core.agents import NewtonVecAgent

    from core.terrain.flat_terrain import FlatBaseTerrainBuilder
    from core.terrain.perlin_terrain import PerlinBaseTerrainBuilder

    from core.sensors import VecIMU
    from core.controllers import VecJointsController
    from core.animation import AnimationEngine
    from core.domain_randomizer import NewtonBaseDomainRandomizer

    imu = VecIMU(
        universe=universe,
        local_position=torch.zeros((num_envs, 3)),
        local_orientation=IDENTITY_QUAT.repeat(num_envs, 1),
        noise_function=lambda x: x,
    )

    joints_constraints = {
        "FR_HAA": (-45, 45),
        "FL_HAA": (-45, 45),
        "HR_HAA": (-45, 45),
        "HL_HAA": (-45, 45),
        "FR_HFE": (-90, 90),
        "HR_HFE": (-90, 90),
        "FL_HFE": (-90, 90),
        "HL_HFE": (-90, 90),
        "FR_KFE": (-180, 180),
        "FL_KFE": (-180, 180),
        "HR_KFE": (-180, 180),
        "HL_KFE": (-180, 180),
    }
    joints_controller = VecJointsController(
        universe=universe,
        joint_constraints=joints_constraints,
        noise_function=lambda x: x,
    )

    newton_agent = NewtonVecAgent(
        num_agents=num_envs,
        imu=imu,
        joints_controller=joints_controller,
    )

    animation_engine = AnimationEngine(
        clips=animation_clips_config,
        step_dt=control_step_dt,
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
            terrain_builders=[FlatBaseTerrainBuilder()],
            domain_randomizer=domain_randomizer,
            inverse_control_frequency=rl_config["newton"]["inverse_control_frequency"],
        )

        env.construct(universe)
        env.reset()  # called manually, because the task usually does it, must be done before stepping

        animation_engine.construct(current_animation)

        joints_names = joints_controller.art_view.joint_names

        # this is very specific to Newton, because we know that it takes joint positions and the animation engine
        # provides that exactly; a different robot or different control mode would probably require a different approach
        while universe.is_playing:
            joint_data = animation_engine.get_current_clip_data_ordered(
                universe.current_time_step_index // control_step_dt,
                joints_names,
            )

            # index 7 is the joint position (angle in degrees)
            joint_positions = joint_data[:, 7]

            # we wrap it in an array to make it 2D (it's a vectorized env)
            env.step(np.array([joint_positions], dtype=np.float32))

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

    # TODO: add a proper separate playing environment
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

    from core.utils.path import build_child_path_with_prefix

    task_runs_directory = "runs"
    task_name = build_child_path_with_prefix(
        rl_config["task_name"], task_runs_directory
    )

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
        save_path=task_name,
    )

    task.construct(universe)

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
            tensorboard_log=task_runs_directory,
        )

        if cli_args.checkpoint is not None:
            model = PPO.load(cli_args.checkpoint, task, device=rl_config["device"])

        model.learn(
            total_timesteps=rl_config["timesteps_per_env"] * num_envs,
            tb_log_name=task_name,
            reset_num_timesteps=False,
            progress_bar=True,
            callback=callback,
        )
        model.save(f"{task_runs_directory}/{task_name}_1/model.zip")

        exit(1)

    if playing:
        from core.utils.path import get_folder_from_path

        model = PPO.load(cli_args.checkpoint)

        actions = model.predict(task.reset()[0], deterministic=True)[0]
        actions = np.array([actions])  # make sure we have a 2D tensor

        log_file = open(f"{get_folder_from_path(cli_args.checkpoint)}/playing.csv", "w")
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
    model = PPO.load(cli_args.checkpoint, device="cpu")

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
        f"{cli_args.checkpoint}.onnx",
        verbose=True,
        input_names=["observations"],
        output_names=["actions"],
    )  # outputs are mu (actions), sigma, value

    print(f"Exported to {cli_args.checkpoint}.onnx!")


if __name__ == "__main__":
    main()
