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
        self.best_mean_reward: float = -np.inf
        self.cumulative_reward: Tensor = torch.zeros((1,))

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
            self.cumulative_reward = np.where(
                task.dones_buf,
                np.zeros_like(self.cumulative_reward),
                self.cumulative_reward + task.rewards_buf.mean().item(),
            )

        if (
            can_check
            and can_save
            and self.cumulative_reward.mean().item() > self.best_mean_reward
        ):
            self.best_mean_reward = self.cumulative_reward.mean().item()
            self.model.save(
                f"{self.logger.dir}/{self.save_path}_rew_{self.best_mean_reward:.2f}"
            )

        self.logger.record(
            "rewards/cumulative",
            self.cumulative_reward.mean().item(),
        )
        self.logger.record("rewards/best_mean", self.best_mean_reward)

        # TODO: better metrics about the agent's state & the animation engine
        agent_observations = task.agent.get_observations()
        self.logger.record(
            "observations/positions",
            np.linalg.norm(agent_observations["positions"].mean()),
        )
        self.logger.record(
            "observations/linear_accelerations",
            np.linalg.norm(agent_observations["linear_accelerations"].mean()),
        )
        self.logger.record(
            "observations/linear_velocities",
            np.linalg.norm(agent_observations["linear_velocities"].mean()),
        )
        self.logger.record(
            "observations/angular_velocities",
            np.linalg.norm(agent_observations["angular_velocities"].mean()),
        )
        self.logger.record(
            "observations/projected_gravities",
            np.linalg.norm(agent_observations["projected_gravities"].mean()),
        )

        return True


class NewtonBaseTask(BaseTask):
    def __init__(
        self,
        training_env: NewtonBaseEnv,
        playing_env: NewtonBaseEnv,
        agent: NewtonBaseAgent,
        num_envs: int,
        device: str,
        playing: bool,
        max_episode_length: int,
        observation_space: Space,
        action_space: Box,
        reward_space: Box,
    ):

        super().__init__(
            training_env,
            playing_env,
            agent,
            num_envs,
            device,
            playing,
            max_episode_length,
            observation_space,
            action_space,
            reward_space,
        )

        self.training_env: NewtonBaseEnv = training_env
        self.playing_env: NewtonBaseEnv = playing_env
        self.agent: NewtonBaseAgent = agent

        self.animation_engine: AnimationEngine = AnimationEngine()  # TODO
        self.last_actions_buf: Actions = np.zeros(
            (self.num_envs, self.num_actions), dtype=np.float32
        )

    @abstractmethod
    def construct(self, universe: Universe) -> None:
        super().construct(universe)

        if self.playing:
            self.playing_env.construct(universe)
        else:
            self.training_env.construct(universe)

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        self.progress_buf += 1

        return super().step_wait()

    @abstractmethod
    def reset(self) -> VecEnvObs:
        super().reset()

        return {}
