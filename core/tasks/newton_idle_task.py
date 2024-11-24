import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

import numpy as np
from core.agents import NewtonBaseAgent
from core.envs import NewtonBaseEnv
from core.tasks import NewtonBaseTask, NewtonBaseTaskCallback
from gymnasium.spaces import Box


class NewtonIdleTaskCallback(NewtonBaseTaskCallback):
    def __init__(self):
        super().__init__()

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
        num_envs: int,
        device: str,
        headless: bool,
        playing: bool,
        max_episode_length: int,
    ):
        self.observation_space: Box = Box(
            low=np.array([-1.0] * 3 + [-np.Inf] * 7),
            high=np.array([1.0] * 3 + [np.Inf] * 7),
        )

        self.action_space: Box = Box(
            low=np.array([-15, -90, -45] * 4),
            high=np.array([15, 90, 180] * 4),
        )

        self.reward_space: Box = Box(
            low=np.array([-2.0]),
            high=np.array([1.0]),
        )

        super().__init__(
            training_env,
            playing_env,
            agent,
            num_envs,
            device,
            headless,
            playing,
            max_episode_length,
            self.observation_space,
            self.action_space,
            self.reward_space,
        )

        self.min_height: float = 0.2

    def construct(self) -> None:
        super().construct()

    def step_wait(self) -> VecEnvStepReturn:
        super().step_wait()

        self.env.step(self.actions_buf, self.headless)

        obs = self._get_observations()
        heights = obs["positions"][:, 2]

        obs_buf: np.ndarray = np.zeros(
            (self.num_envs, self.num_observations), dtype=np.float32
        )
        obs_buf[:, :3] = obs["projected_gravities"]
        obs_buf[:, 3:6] = obs["linear_accelerations"]
        obs_buf[:, 6:9] = obs["angular_velocities"]
        obs_buf[:, 9] = heights

        self._calculate_rewards()

        # terminated
        self.dones_buf = np.where(heights <= self.min_height, True, False)
        self.dones_buf = np.where(heights >= 2.0, True, self.dones_buf)

        self.dones_buf = np.where(
            self.progress_buf >= self.max_episode_length,
            True,
            self.dones_buf,
        )  # truncated

        # creates a new np array with only the indices of the environments that are done
        resets = self.dones_buf.nonzero()[0].flatten()
        if len(resets) > 0:
            self.env.reset(resets)

        # clears the last 2 observations & the progress if any Newton is reset
        obs_buf[resets, :] = 0.0
        self.progress_buf[resets] = 0

        self.last_actions_buf = self.actions_buf.copy()

        return (
            obs_buf.copy(),
            self.rewards_buf.copy(),
            self.dones_buf.copy(),
            self.infos_buf,
        )

    def reset(self) -> VecEnvObs:
        super().reset()

        self.env.reset()

        obs = self._get_observations()

        obs_buf: np.ndarray = np.zeros(
            (self.num_envs, self.num_observations), dtype=np.float32
        )

        obs_buf[:, :3] = obs["projected_gravities"]
        obs_buf[:, 3:6] = obs["linear_accelerations"]
        obs_buf[:, 6:9] = obs["angular_velocities"]
        obs_buf[:, 9] = obs["positions"][:, 2]

        return obs_buf

    def _get_observations(self) -> VecEnvObs:
        env_observations = self.env.get_observations()

        return env_observations

    def _calculate_rewards(self) -> None:
        obs = self._get_observations()
        positions = obs["positions"]
        angular_velocities = obs["angular_velocities"]
        linear_velocities = obs["linear_velocities"]
        joint_accelerations = (
            self.agent.newton_art_view.get_applied_joint_efforts().cpu().numpy()
        )

        heights = positions[:, 2]

        linear_velocity_xy_reward = (
            np.exp(-np.sum(np.square(linear_velocities[:, :2])) / 0.25) * 1.0
        )
        linear_velocity_z_reward = (
            np.exp(-np.sum(np.square(linear_velocities[:, 2])) / 0.25) * -0.03
        )
        angular_velocity_z_reward = (
            np.exp(-np.sum(np.square(angular_velocities[:, 2])) / 0.25) * 0.5
        )

        action_rate_reward = (
            np.sum(np.square(self.actions_buf - self.last_actions_buf)) * 0.2
        ) * -0.006

        joint_acceleration_reward = (
            np.exp(-np.sum(np.square(joint_accelerations)) / 0.25) * -0.0003
        )

        self.rewards_buf = (
            linear_velocity_xy_reward
            + linear_velocity_z_reward
            + angular_velocity_z_reward
            + action_rate_reward
            + joint_acceleration_reward
        )

        self.rewards_buf = np.where(heights <= self.min_height, -2.0, self.rewards_buf)
        self.rewards_buf = np.where(heights >= 2.0, -2.0, self.rewards_buf)

        # TODO: should this be here?
        self.rewards_buf = np.clip(
            self.rewards_buf,
            self.reward_space.low,
            self.reward_space.high,
        )
