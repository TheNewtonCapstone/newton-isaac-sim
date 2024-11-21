from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

import numpy as np
from core.agents import NewtonBaseAgent
from core.envs import NewtonBaseEnv
from core.tasks import NewtonBaseTask, NewtonBaseTaskCallback
from gymnasium.spaces import Box
from torch import Tensor


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
            low=np.array([-1.0] * 3 + [-np.Inf] * 1),
            high=np.array([1.0] * 3 + [np.Inf] * 1),
        )

        self.action_space: Box = Box(
            low=np.array([-1.0] * 12),
            high=np.array([1.0] * 12),
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

    def construct(self) -> None:
        super().construct()

    def step_wait(self) -> VecEnvStepReturn:
        super().step_wait()

        self.env.step(self.actions_buf * 360, self.headless)

        obs = self._get_observations()

        obs_buf: np.ndarray = np.zeros(
            (self.num_envs, self.num_observations), dtype=np.float32
        )
        obs_buf[:, :3] = obs["projected_gravities"]
        obs_buf[:, 3] = obs["positions"][:, 2]

        self._calculate_rewards()

        return (
            obs_buf.copy(),
            self.rewards_buf.copy(),
            self.dones_buf.copy(),
            self.infos_buf,
        )

    def reset(self) -> VecEnvObs:
        super().reset()

        obs = self._get_observations()

        reset_buf: np.ndarray = np.zeros(
            (self.num_envs, self.num_observations), dtype=np.float32
        )

        reset_buf[:, :3] = obs["projected_gravities"]
        reset_buf[:, 3] = obs["positions"][:, 2]

        return reset_buf

    def _get_observations(self) -> VecEnvObs:
        env_observations = self.env.get_observations()

        return env_observations

    def _calculate_rewards(self) -> None:
        obs = self._get_observations()
        positions = obs["positions"]
        projected_gravities = obs["projected_gravities"]

        heights = positions[:, 2]

        # normalize gravity and projected gravity
        gravity = np.array(self.env.world.physics_sim_view.get_gravity())
        normalized_gravity = gravity / np.linalg.norm(gravity)
        normalized_gravities = np.repeat(
            normalized_gravity[np.newaxis, :], len(positions), axis=0
        )

        normalized_projected_gravities = (
            projected_gravities
            / np.linalg.norm(projected_gravities, axis=1)[:, np.newaxis]
        )

        self.rewards_buf = (
            np.sum(normalized_gravities * normalized_projected_gravities, axis=1)
            * heights
        )

        # TODO: should this be here?
        self.rewards_buf = np.clip(
            self.rewards_buf,
            self.reward_space.low,
            self.reward_space.high,
        )
