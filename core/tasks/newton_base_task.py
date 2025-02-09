from abc import abstractmethod
from typing import Optional

import numpy as np
import torch as th
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from core.agents import NewtonBaseAgent
from core.animation import AnimationEngine
from core.envs.newton_base_env import NewtonBaseEnv
from core.tasks.base_task import BaseTask, BaseTaskCallback
from core.types import Actions
from core.universe import Universe
from gymnasium import Space
from gymnasium.spaces import Box
from ..archiver.archiver import Config


class NewtonBaseTaskCallback(BaseTaskCallback):
    def __init__(self, check_freq: int, save_path: str):
        super().__init__()

        self.check_freq: int = check_freq
        self.save_path: str = save_path
        self.best_median_reward: float = -np.inf
        self.cumulative_rewards: th.Tensor = th.zeros((0,))

    def _init_callback(self) -> None:
        super()._init_callback()

        task: NewtonBaseTask = self.training_env

        if self.save_path is not None:
            self.model.save(f"{self.logger.dir}/{self.save_path}")

        self.cumulative_rewards = th.zeros((task.num_envs,), device=task.device)

    def _on_step(self) -> bool:
        super()._on_step()

        task: NewtonBaseTask = self.training_env

        can_check = self.n_calls > self.check_freq
        can_save = self.n_calls % self.check_freq == 0

        # Saves best cumulative rewards
        if can_check:
            self.cumulative_rewards = th.where(
                task.dones_buf,
                th.zeros_like(task.rewards_buf),
                self.cumulative_rewards + task.rewards_buf,
            )

        if (
            can_check
            and can_save
            and self.cumulative_rewards.median().item() > self.best_median_reward
        ):
            self.best_median_reward = self.cumulative_rewards.median().item()
            self.logger.record("rewards/best_median", self.best_median_reward)

            self.model.save(
                f"{self.logger.dir}/{self.save_path}_rew_{self.best_median_reward:.2f}"
            )

        # TODO: better metrics about the agent's state & the animation engine
        env_observations = task.env.get_observations()

        for k, v in env_observations.items():
            # we really only care about the z component of gravity
            if "gravities" in k:
                self.logger.record(f"observations/{k}", v[:, -1].mean().item())
                continue

            # if it's a bool (i.e. contact data), we want to know the percentage of contacts
            if v.dtype == th.bool:
                self.logger.record(
                    f"observations/{k}",
                    v.to(dtype=th.float32).mean(dim=-1).mean().item(),
                )
                continue

            self.logger.record(
                f"observations/{k}",
                th.linalg.norm(v, dim=-1).mean().item(),
            )

        return True


class NewtonBaseTask(BaseTask):
    def __init__(
        self,
        universe: Universe,
        name: str,
        env: NewtonBaseEnv,
        agent: NewtonBaseAgent,
        animation_engine: AnimationEngine,
        num_envs: int,
        device: str,
        playing: bool,
        reset_in_play: bool,
        max_episode_length: int,
        observation_space: Space,
        action_space: Box,
        reward_space: Box,
        dr_configurations: Config,
    ):

        super().__init__(
            universe,
            name,
            env,
            agent,
            num_envs,
            device,
            playing,
            reset_in_play,
            max_episode_length,
            observation_space,
            action_space,
            reward_space,
            dr_configurations,
        )

        self.training_env: NewtonBaseEnv = env
        self.agent: NewtonBaseAgent = agent

        self.animation_engine: AnimationEngine = animation_engine
        self.air_time: th.Tensor = th.zeros(
            (self.num_envs, 4),
            device=self.device,
        )  # air time per paw
        self.last_actions_buf: Actions = th.zeros(
            (2, self.num_envs, self.num_actions),
            dtype=th.float32,
            device=self.device,
        )  # 2 sets of past actions, 0: t - 1, 1: t - 2

    @abstractmethod
    def construct(self) -> None:
        super().construct()

        self.env.register_self()

        self.animation_engine.register_self(self.name)

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        self.progress_buf += 1

        return super().step_wait()

    @abstractmethod
    def reset(self) -> VecEnvObs:
        return super().reset()
