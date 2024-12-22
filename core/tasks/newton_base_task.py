from abc import abstractmethod

from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

import numpy as np
import torch
from core.agents import NewtonBaseAgent
from core.animation import AnimationEngine
from core.envs.newton_base_env import NewtonBaseEnv
from core.tasks.base_task import BaseTask, BaseTaskCallback
from core.types import Actions
from core.universe import Universe
from gymnasium import Space
from gymnasium.spaces import Box
from torch import Tensor


class NewtonBaseTaskCallback(BaseTaskCallback):
    def __init__(self, check_freq: int, save_path: str):
        super().__init__()

        self.check_freq: int = check_freq
        self.save_path: str = save_path
        self.best_median_reward: float = -np.inf
        self.cumulative_rewards: Tensor = torch.zeros((0,))

    def _init_callback(self) -> None:
        if self.save_path is not None:
            self.model.save(f"{self.logger.dir}/{self.save_path}")

    def _on_step(self) -> bool:
        super()._on_step()

        task: NewtonBaseTask = self.training_env

        can_check = self.n_calls > self.check_freq
        can_save = self.n_calls % self.check_freq == 0

        # Saves best cumulative rewards
        if can_check:
            self.cumulative_rewards = torch.where(
                task.dones_buf,
                torch.zeros_like(task.rewards_buf),
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
            if v.dtype == torch.bool:
                self.logger.record(
                    f"observations/{k}",
                    v.to(dtype=torch.float32).mean(dim=-1).mean().item(),
                )
                continue

            self.logger.record(
                f"observations/{k}",
                torch.linalg.norm(v, dim=-1).mean().item(),
            )

        return True


class NewtonBaseTask(BaseTask):
    def __init__(
        self,
        name: str,
        training_env: NewtonBaseEnv,
        playing_env: NewtonBaseEnv,
        agent: NewtonBaseAgent,
        animation_engine: AnimationEngine,
        num_envs: int,
        device: str,
        playing: bool,
        max_episode_length: int,
        rl_step_dt: float,
        observation_space: Space,
        action_space: Box,
        reward_space: Box,
    ):

        super().__init__(
            name,
            training_env,
            playing_env,
            agent,
            num_envs,
            device,
            playing,
            max_episode_length,
            rl_step_dt,
            observation_space,
            action_space,
            reward_space,
        )

        self.training_env: NewtonBaseEnv = training_env
        self.playing_env: NewtonBaseEnv = playing_env
        self.agent: NewtonBaseAgent = agent

        self.animation_engine: AnimationEngine = animation_engine
        self.last_actions_buf: Actions = torch.zeros(
            (2, self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )  # 2 sets of past actions, 0: t - 1, 1: t - 2

    @abstractmethod
    def construct(self, universe: Universe) -> None:
        super().construct(universe)

        if self.playing:
            self.playing_env.construct(universe)
        else:
            self.training_env.construct(universe)

        self.animation_engine.construct(self.name)

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        self.progress_buf += 1

        return super().step_wait()

    @abstractmethod
    def reset(self) -> VecEnvObs:
        super().reset()

        return {}
