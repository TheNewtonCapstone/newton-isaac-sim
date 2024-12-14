import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

import numpy as np
from core.agents import NewtonBaseAgent
from core.animation import AnimationEngine
from core.envs import NewtonBaseEnv
from core.tasks import NewtonBaseTask, NewtonBaseTaskCallback
from core.universe import Universe
from gymnasium.spaces import Box


class NewtonIdleTaskCallback(NewtonBaseTaskCallback):
    def __init__(self, check_freq: int, save_path: str):
        super().__init__(check_freq, save_path)

        self.training_env: NewtonIdleTask

    def _on_step(self) -> bool:
        super()._on_step()

        # TODO: metrics about the agent's state & the animation engine

        return True


class NewtonIdleTask(NewtonBaseTask):
    def __init__(
        self,
        training_env: NewtonBaseEnv,
        playing_env: NewtonBaseEnv,
        agent: NewtonBaseAgent,
        animation_engine: AnimationEngine,
        num_envs: int,
        device: str,
        playing: bool,
        max_episode_length: int,
        rl_step_dt: float,
    ):
        self.observation_space: Box = Box(
            low=np.array(
                [-10.0] * 3  # for the projected gravity
                + [-np.Inf] * 6  # for linear & angular velocities
                + [-1.0] * 12  # for the joint positions
                + [-np.Inf] * 12  # for the joint velocities
                + [-1.0] * 24  # for the previous 2 actions
                + [-1.0] * 2,  # for the transformed phase signal
            ),
            high=np.array(
                [10.0] * 3  # for the projected gravity
                + [np.Inf] * 6  # for linear & angular velocities
                + [1.0] * 12  # for the joint positions
                + [np.Inf] * 12  # for the joint velocities
                + [1.0] * 24  # for the previous 2 actions
                + [1.0] * 2  # for the transformed phase signal,
            ),
        )

        self.action_space: Box = Box(
            low=np.array([-1.0] * 12),
            high=np.array([1.0] * 12),
        )

        self.reward_space: Box = Box(
            low=np.array([-np.Inf]),
            high=np.array([np.Inf]),
        )

        super().__init__(
            "newton_idle",
            training_env,
            playing_env,
            agent,
            animation_engine,
            num_envs,
            device,
            playing,
            max_episode_length,
            rl_step_dt,
            self.observation_space,
            self.action_space,
            self.reward_space,
        )

        self.reset_height: float = 0.1

    def construct(self, universe: Universe) -> None:
        super().construct(universe)

    def step_wait(self) -> VecEnvStepReturn:
        super().step_wait()

        self.env.step(self.actions_buf)

        # TODO: Integrate the animation engine into the Newton*Tasks
        #   Update the observations & reward function accordingly
        #   From what I currently understand, we add two observations to the observation space:
        #    - The phase signal (0-1) of the phase progress, transformed by cos(2*pi*progress) & sin(2*pi*progress)
        #   This would allow the model to recognize the periodicity of the animation and hopefully learn the intricacies
        #   of each gait.
        #   The reward function would compare desired joint positions with the current joint positions, and reward
        #   the agent for getting closer to the desired positions.

        obs_buf = self._get_observations()
        self._calculate_rewards()

        # only now can we save the actions (after gathering observations & rewards)
        self.last_actions_buf[1, :, :] = self.last_actions_buf[0, :, :]
        self.last_actions_buf[0, :, :] = self.actions_buf.copy()

        gravity_on_z = self.env.get_observations()["projected_gravities"][:, 2]
        # terminated
        self.dones_buf = np.where(
            gravity_on_z >= 0.0, True, False
        )  # when we're not upright

        self.dones_buf = np.where(
            self.progress_buf > self.max_episode_length,
            not self.playing,
            self.dones_buf,
        )  # truncated

        # creates a new np array with only the indices of the environments that are done
        resets = self.dones_buf.nonzero()[0].flatten()
        if len(resets) > 0:
            self.env.reset(resets)

        # clears the last 2 observations & the progress if any Newton is reset
        obs_buf[resets, :] = 0.0
        self.progress_buf[resets] = 0
        self.last_actions_buf[:, resets, :] = 0.0

        return (
            obs_buf.copy(),
            self.rewards_buf.copy(),
            self.dones_buf.copy(),
            self.infos_buf,
        )

    def reset(self) -> VecEnvObs:
        super().reset()

        obs_buf = self._get_observations()

        self.env.reset()

        # we want to return the last observation of the previous episode, according to the STB3 documentation
        return obs_buf

    def _get_observations(self) -> VecEnvObs:
        obs = self.env.get_observations()

        phase_signal = self.progress_buf / self.max_episode_length

        obs_buf: np.ndarray = np.zeros(
            (self.num_envs, self.num_observations), dtype=np.float32
        )
        obs_buf[:, :3] = obs["projected_gravities"]
        obs_buf[:, 3:6] = obs["linear_velocities"]
        obs_buf[:, 6:9] = obs["angular_velocities"]
        obs_buf[:, 9:21] = (
            self.agent.joints_controller.get_normalized_joint_positions().cpu().numpy()
        )
        obs_buf[:, 21:33] = (
            self.agent.joints_controller.get_joint_velocities_deg().cpu().numpy()
        )
        # 1st & 2nd set of past actions, we don't care about just-applied actions
        obs_buf[:, 33:57] = self.last_actions_buf[:].reshape(
            (self.num_envs, self.num_actions * 2)
        )

        # From what I currently understand, we add two observations to the observation space:
        #  - The phase signal (0-1) of the phase progress, transformed by cos(2*pi*progress) & sin(2*pi*progress)
        # This would allow the model to recognize the periodicity of the animation and hopefully learn the intricacies
        # of each gait.
        obs_buf[:, 57] = np.cos(2 * np.pi * phase_signal)
        obs_buf[:, 58] = np.sin(2 * np.pi * phase_signal)

        return obs_buf

    def _calculate_rewards(self) -> None:
        # TODO: rework rewards for Newton*Tasks
        #   The current reward function is missing many features, all in comments below

        obs = self.env.get_observations()
        angular_velocities = obs["angular_velocities"]
        linear_velocities = obs["linear_velocities"]
        projected_gravities = obs["projected_gravities"]

        joints_order = self.agent.joints_controller.art_view.joint_names

        # base position
        # base orientation
        base_linear_velocity_xy = linear_velocities[:, :2]
        base_linear_velocity_z = linear_velocities[:, 2]
        base_angular_velocity_xy = angular_velocities[:, :2]
        base_angular_velocity_z = angular_velocities[:, 2]
        joint_positions = (
            self.agent.joints_controller.get_normalized_joint_positions().cpu().numpy()
        )  # [-1, 1] unitless
        joint_velocities = (
            self.agent.joints_controller.get_normalized_joint_velocities().cpu().numpy()
        )  # [-1, 1] unitless
        # joint_accelerations
        joint_efforts = (
            self.agent.joints_controller.art_view.get_measured_joint_efforts()
            .cpu()
            .numpy()
        )  # in Nm
        last_animation_joint_data = (
            self.animation_engine.get_current_clip_datas_ordered(
                np.where(
                    self.progress_buf == 0,
                    self.max_episode_length - 1,
                    self.progress_buf - 1,
                ),
                joints_order,
            )
        )
        animation_joint_data = self.animation_engine.get_current_clip_datas_ordered(
            self.progress_buf,
            joints_order,
        )
        # we use the joint controller here, because it contains all the required information
        animation_joint_positions = (
            self.agent.joints_controller.normalize_joint_positions(
                torch.from_numpy(animation_joint_data[:, :, 7])
            )
            .cpu()
            .numpy()
        )  # [-1, 1] unitless
        animation_joint_velocities = (
            self.agent.joints_controller.normalize_joint_positions(
                torch.from_numpy(
                    (animation_joint_data[:, :, 7] - last_animation_joint_data[:, :, 7])
                    / self.rl_step_dt
                )
            )
            .cpu()
            .numpy()
        )  # [-1, 1] unitless

        from core.utils.rl import (
            exp_squared_norm,
            fd_first_order_squared_norm,
            fd_second_order_squared_norm,
        )

        # base position reward
        # base orientation reward
        base_linear_velocity_xy_reward = exp_squared_norm(
            base_linear_velocity_xy,
            mult=-8.0,
            weight=1.0,
        )
        base_linear_velocity_z_reward = exp_squared_norm(
            base_angular_velocity_z,
            mult=-8.0,
            weight=1.0,
        )
        base_angular_velocity_xy_reward = exp_squared_norm(
            base_angular_velocity_xy,
            mult=-2,
            weight=0.5,
        )
        base_angular_velocity_z_reward = exp_squared_norm(
            base_angular_velocity_z,
            mult=-2,
            weight=0.5,
        )
        joint_positions_reward = fd_first_order_squared_norm(
            joint_positions,
            animation_joint_positions,
            weight=15.0,
        )
        joint_velocities_reward = fd_first_order_squared_norm(
            joint_velocities,
            animation_joint_velocities,
            weight=0.01,
        )
        joint_efforts_reward = fd_first_order_squared_norm(
            joint_efforts,
            np.zeros_like(joint_efforts),
            weight=0.001,
        )
        # joint accelerations reward
        joint_action_rate_reward = fd_first_order_squared_norm(
            self.actions_buf,
            self.last_actions_buf[0],
            weight=1.0,
        )
        joint_action_acceleration_reward = fd_second_order_squared_norm(
            self.actions_buf,
            self.last_actions_buf[0],
            self.last_actions_buf[1],
            weight=0.45,
        )

        self.rewards_buf = (
            base_linear_velocity_xy_reward
            + base_linear_velocity_z_reward
            + base_angular_velocity_xy_reward
            + base_angular_velocity_z_reward
            + joint_positions_reward
            + joint_velocities_reward
            + joint_efforts_reward
            + joint_action_rate_reward
            + joint_action_acceleration_reward
        )

        gravity_on_z = projected_gravities[:, 2]
        self.rewards_buf = np.where(
            gravity_on_z >= 0.0, self.rewards_buf - 10.0, self.rewards_buf
        )

        # TODO: should this be here?
        self.rewards_buf = np.clip(
            self.rewards_buf,
            self.reward_space.low,
            self.reward_space.high,
        )
