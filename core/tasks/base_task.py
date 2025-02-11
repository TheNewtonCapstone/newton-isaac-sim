from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Type

import numpy as np
import torch
from gymnasium.core import RenderFrame
from rsl_rl.env import VecEnv
from skrl.envs.torch import Wrapper
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium
from ..agents import BaseAgent
from ..archiver import Archiver
from ..base import BaseObject
from ..envs import BaseEnv
from ..types import EpisodeLength, Rewards, Dones, Actions, Extras, Observations, StepReturn, ResetReturn, \
    TaskObservations
from ..universe import Universe


class BaseTaskCallback(BaseCallback):
    def _init_callback(self) -> None:
        task: BaseTask = self.training_env

        self.logger.record("meta/name", task.name)
        self.logger.record("meta/agent/name", task.agent.__class__.__name__)

        self.logger.record("meta/device", task.device)

        self.logger.record("meta/num_envs", task.num_envs)
        self.logger.record("meta/max_episode_length", task.max_episode_length)
        self.logger.record("meta/rl_step_dt", task._universe.control_dt)

        self.logger.record("meta/observation_space", task.observation_space)
        self.logger.record("meta/num_observations", task.num_obs)
        self.logger.record("meta/action_space", task.action_space)
        self.logger.record("meta/num_actions", task.num_actions)
        self.logger.record("meta/reward_space", task.reward_space)

    def _on_step(self) -> bool:
        task: BaseTask = self.training_env

        median_dones: float = torch.median(task.reset_buf.sum(dim=-1)).item()
        median_reward: float = torch.median(task.rew_buf).item()
        mean_progress: float = task.episode_length_buf.mean().item()
        mean_action: float = task.actions_buf.mean().item()

        self.logger.record("dones/median", median_dones)
        self.logger.record("rewards/median", median_reward)
        self.logger.record("progress/mean", mean_progress)
        self.logger.record("actions/mean", mean_action)

        task_archive_data = {
            "dones_median": median_dones,
            "rewards_median": median_reward,
            "progress_mean": mean_progress,
            "actions_mean": mean_action,
        }
        Archiver.put("rl", task_archive_data)

        return True


class BaseTask(BaseObject, gymnasium.vector.VectorEnv):
    def __init__(
            self,
            universe: Universe,
            name: str,
            env: BaseEnv,
            agent: BaseAgent,
            num_envs: int,
            device: str,
            playing: bool,
            reset_in_play: bool,
            max_episode_length: int,
            observation_space: gymnasium.spaces.Space,
            action_space: gymnasium.spaces.Box,
            reward_space: gymnasium.spaces.Box,
    ):
        BaseObject.__init__(
            self,
            universe=universe,
        )

        # We type hint universe again here to avoid circular imports
        self._universe: Universe = universe

        self.register_self()

        self.name: str = name
        self.device: str = device
        self.playing: bool = playing
        self.reset_in_play: bool = reset_in_play

        self.agent: BaseAgent = agent
        self.env: BaseEnv = env

        self.observation_space: gymnasium.spaces.Space = observation_space
        self.action_space: gymnasium.spaces.Box = action_space
        self.reward_space: gymnasium.spaces.Box = reward_space

        self.num_envs: int = num_envs
        self.num_agents: int = 1
        self.max_episode_length: int = max_episode_length

        self.num_privileged_obs: int = 0  # unused
        self.num_obs: int = self.observation_space.shape[0]
        self.num_actions: int = self.action_space.shape[0]

        self.closed: bool = False

        self.obs_buf: Observations = torch.zeros(
            (self.num_envs, self.num_obs),
            dtype=torch.float32,
            device=self.device,
        )
        self.actions_buf: Actions = torch.zeros(
            (self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.rew_buf: Rewards = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.reset_buf: Dones = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.episode_length_buf: EpisodeLength = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.extras: Extras = {"observations": {}}

        VecEnv.__init__(self)

    def __repr__(self):
        return f"BaseTask: {self.num_envs} environments, {self.num_obs} observations, {self.num_actions} actions"

    @abstractmethod
    def construct(self) -> None:
        BaseObject.construct(self)

    @abstractmethod
    def post_construct(self):
        BaseObject.post_construct(self)

    @abstractmethod
    def step(self, actions: Actions) -> StepReturn:
        assert self._is_post_constructed, "Task not constructed: tried to step"

        self.actions_buf = actions

        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.reset_buf,
            self.extras,
        )

    @abstractmethod
    def reset(self) -> ResetReturn:
        assert self._is_post_constructed, "Task not constructed: tried to reset"

        return self.obs_buf, self.extras

    @abstractmethod
    def get_observations(self) -> TaskObservations:
        return self.obs_buf, self.extras

    def render(self) -> tuple[RenderFrame, ...] | None:
        pass