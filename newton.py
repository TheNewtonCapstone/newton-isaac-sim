import argparse
import copy

import numpy as np
import torch
from core.utils.config import load_config
from core.utils.path import (
    build_child_path_with_prefix,
    get_folder_from_path,
)
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy


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
        default="configs/newton_idle_task.yaml",
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


if __name__ == "__main__":
    parser = setup_argparser()

    cli_args = parser.parse_args()
    rl_config = load_config(cli_args.rl_config)
    world_config = load_config(cli_args.world_config)
    randomization_config = load_config(cli_args.randomization_config)

    physics_only = cli_args.physics
    exporting = cli_args.export_onnx
    pidding = not exporting and cli_args.pid_control
    playing = not exporting and not pidding and cli_args.play
    training = not exporting and not pidding and not cli_args.play
    headless = (
        cli_args.headless or cli_args.export_onnx
    )  # if we're exporting, don't show the GUI

    # override config with CLI num_envs, if specified
    if cli_args.num_envs != -1:
        rl_config["n_envs"] = cli_args.num_envs
    elif not training:
        rl_config["n_envs"] = 1

    if playing:
        # increase the number of steps if we're playing
        rl_config["ppo"]["n_steps"] *= 4

    if playing or pidding:
        # force the physics device to CPU if we're playing or pidding (for better control)
        world_config["device"] = "cpu"

    # ensure proper config reading when encountering None
    if rl_config["ppo"]["clip_range_vf"] == "None":
        rl_config["ppo"]["clip_range_vf"] = None

    if rl_config["ppo"]["target_kl"] == "None":
        rl_config["ppo"]["target_kl"] = None

    print(
        f"Running with {rl_config['n_envs']} environments, {rl_config['ppo']['n_steps']} steps per environment, and {'headless' if headless else 'GUI'} mode.\n",
        f"{'Exporting ONNX' if exporting else 'Playing' if playing else 'Training' if training else 'Pidding'}.\n",
        f"Using {rl_config['device']} as the RL device and {world_config['device']} as the physics device.",
    )

    from isaacsim import SimulationApp

    sim_app = SimulationApp(
        {"headless": headless}, experience="./apps/omni.isaac.sim.newton.kit"
    )

    from core.envs import NewtonMultiTerrainEnv
    from core.agents import NewtonVecAgent

    from core.terrain.flat_terrain import FlatTerrainBuilder
    from core.terrain.perlin_terrain import PerlinTerrainBuilder

    # ---------- #
    # SIMULATION #
    # ---------- #

    if physics_only:
        newton = NewtonVecAgent(num_agents=rl_config["n_envs"])

        env = NewtonMultiTerrainEnv(
            agent=newton,
            num_envs=rl_config["n_envs"],
            terrain_builders=[PerlinTerrainBuilder(), FlatTerrainBuilder()],
            world_settings=world_config,
            randomizer_settings=randomization_config,
        )

        env.construct()
        env.reset()

        while sim_app.is_running() and env.world.is_playing():
            env.step(np.zeros((rl_config["n_envs"], 12)), render=not headless)

        exit(1)

    # ----------- #
    #     RL      #
    # ----------- #

    from core.tasks import NewtonIdleTask, NewtonBaseTaskCallback
    from core.envs import NewtonMultiTerrainEnv
    from core.wrappers import RandomDelayWrapper

    newton_agent = NewtonVecAgent(num_agents=rl_config["n_envs"])

    terrains_size = torch.tensor([10, 10])
    terrains_resolution = torch.tensor([20, 20])

    training_env = NewtonMultiTerrainEnv(
        agent=newton_agent,
        num_envs=rl_config["n_envs"],
        world_settings=world_config,
        terrain_builders=[
            FlatTerrainBuilder(size=terrains_size),
            PerlinTerrainBuilder(
                size=terrains_size,
                resolution=terrains_resolution,
                height=0.05,
                octave=4,
                noise_scale=2,
            ),
            PerlinTerrainBuilder(
                size=terrains_size,
                resolution=terrains_resolution,
                height=0.03,
                octave=8,
                noise_scale=4,
            ),
            PerlinTerrainBuilder(
                size=terrains_size,
                resolution=terrains_resolution,
                height=0.02,
                octave=16,
                noise_scale=8,
            ),
        ],
        randomizer_settings=randomization_config,
    )

    # TODO: add a separate playing environment
    playing_env = copy.deepcopy(training_env)

    task_runs_directory = "runs"
    task_name = build_child_path_with_prefix(
        rl_config["task_name"], task_runs_directory
    )

    # task used for either training or playing
    task = NewtonIdleTask(
        training_env=training_env,
        playing_env=playing_env,
        agent=newton_agent,
        headless=cli_args.headless,
        device=rl_config["device"],
        num_envs=rl_config["n_envs"],
        playing=playing,
        max_episode_length=rl_config["ppo"]["n_steps"],
    )
    callback = NewtonBaseTaskCallback(
        check_freq=64,
        save_path=task_name,
    )

    task.construct()

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
            total_timesteps=rl_config["timesteps_per_env"] * rl_config["n_envs"],
            tb_log_name=task_name,
            reset_num_timesteps=False,
            progress_bar=True,
            callback=callback,
        )
        model.save(f"{task_runs_directory}/{task_name}_1/model.zip")

        exit(1)

    if playing:
        model = PPO.load(cli_args.checkpoint)

        actions = model.predict(task.reset()[0], deterministic=True)[0]
        actions = np.array([actions])  # make sure we have a 2D tensor

        log_file = open(f"{get_folder_from_path(cli_args.checkpoint)}/playing.csv", "w")
        print("time,dt,roll,action1,action2", file=log_file)

        while sim_app.is_running():
            if task.env.world.is_playing():
                step_return = task.step(actions)
                observations = step_return[0]
                actions = model.predict(observations, deterministic=True)[0]

                torque_actions = actions_to_torque(torch.from_numpy(actions[0]))
                print(
                    f"{task.env.world.current_time},{task.env.world.get_physics_dt()},{observations[0][0]},{torque_actions[0]},{torque_actions[1]}",
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
