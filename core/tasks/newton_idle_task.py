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

        self._calculate_rewards(obs["projected_gravities"], obs["positions"])

        return (
            np.concatenate(
                [obs["projected_gravities"], obs["positions"][:, 2]],
                axis=1,
            ),
            self.rewards_buf.copy(),
            self.dones_buf.copy(),
            self.infos_buf,
        )

    def reset(self) -> VecEnvObs:
        super().reset()

        obs = self._get_observations()

        return np.concatenate(
            [obs["projected_gravities"], obs["positions"][:, 2]],
            axis=1,
        )

    def _get_observations(self) -> VecEnvObs:
        env_observations = self.env.get_observations()

        return env_observations

    def _calculate_rewards(
        self, projected_gravities: np.ndarray, positions: np.ndarray
    ) -> None:
        heights = positions[:, 2]

        gravities = np.tile(
            self.env.world.physics_sim_view.get_gravity(),
            (self.num_envs, 1),
        )
        gravities = gravities / np.linalg.norm(gravities)

        normalized_projected_gravities = projected_gravities / np.linalg.norm(
            projected_gravities, axis=1, keepdims=True
        )

        self.rewards_buf = (
            np.sum(gravities * normalized_projected_gravities, axis=1) * heights
        )
        # TODO: should this be here?
        self.rewards_buf = np.clip(
            self.rewards_buf,
            self.reward_space.low,
            self.reward_space.high,
        )
