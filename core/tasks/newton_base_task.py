from abc import abstractmethod
from typing import Optional

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
        super()._init_callback()

        task: NewtonBaseTask = self.training_env

        if self.save_path is not None:
            self.model.save(f"{self.logger.dir}/{self.save_path}")

        self.cumulative_rewards = torch.zeros((task.num_envs,), device=task.device)

    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()

        self._custom_ppo_adaptive_learning_rate()

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

    def _custom_ppo_adaptive_learning_rate(self):
        # support for CustomPPO's implementation of dynamic learning rates

        from core.algorithms import CustomPPO

        if not isinstance(self.model, CustomPPO):
            return

        new_lr = self._get_adaptive_learning_rate(
            self.model.target_kl,
            self.model.current_kl,
            self.model.learning_rate,
        )
        self.model.update_learning_rate(new_lr)

    def _get_adaptive_learning_rate(
        self,
        current_lr: float,
        current_kl: float,
        target_kl: float,
    ) -> float:
        if current_kl > 2 * target_kl:
            return max(1e-5, current_lr / 1.5)

        if current_kl < 0.5 * target_kl:
            return min(1e-2, 1.5 * current_lr)


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
        max_episode_length: int,
        observation_space: Space,
        action_space: Box,
        reward_space: Box,
    ):

        super().__init__(
            universe,
            name,
            env,
            agent,
            num_envs,
            device,
            playing,
            max_episode_length,
            observation_space,
            action_space,
            reward_space,
        )

        self.training_env: NewtonBaseEnv = env
        self.agent: NewtonBaseAgent = agent

        self.animation_engine: AnimationEngine = animation_engine
        self.last_actions_buf: Actions = torch.zeros(
            (2, self.num_envs, self.num_actions),
            dtype=torch.float32,
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
