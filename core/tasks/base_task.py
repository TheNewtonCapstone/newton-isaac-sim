from abc import abstractmethod
from typing import Optional, Tuple

import gymnasium
import torch
from gymnasium.core import RenderFrame
from rsl_rl.env import VecEnv

from ..agents import BaseAgent
from ..base import BaseObject
from ..envs import BaseEnv
from ..types import (
    EpisodeLength,
    Rewards,
    Dones,
    Actions,
    Extras,
    Observations,
    StepReturn,
    ResetReturn,
    TaskObservations,
    ObservationScalers,
    RewardScalers,
    ActionScaler,
    Terminated,
    Truncated,
)
from ..universe import Universe


class BaseTask(BaseObject, VecEnv):
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
        observation_scalers: Optional[ObservationScalers] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scalers: Optional[RewardScalers] = None,
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

        self.observation_scalers: Optional[ObservationScalers] = observation_scalers
        self.action_scaler: Optional[ActionScaler] = action_scaler
        self.reward_scalers: Optional[RewardScalers] = reward_scalers

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
        self.terminated_buf: Terminated = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.truncated_buf: Truncated = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.episode_length_buf: EpisodeLength = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.extras: Extras = {"episode": {}, "time_outs": torch.zeros(self.num_envs)}

        VecEnv.__init__(self)

    def __repr__(self):
        return f"BaseTask: {self.num_envs} environments, {self.num_obs} observations, {self.num_actions} actions"

    @property
    def dones_buf(self) -> Dones:
        return self.terminated_buf | self.truncated_buf

    @property
    def should_reset(self) -> bool:
        return self.reset_in_play or not self.playing

    @abstractmethod
    def construct(self) -> None:
        BaseObject.construct(self)

    @abstractmethod
    def post_construct(self):
        BaseObject.post_construct(self)

    @abstractmethod
    def step(self, actions: Actions) -> StepReturn:
        assert self._is_post_constructed, "Task not constructed: tried to step"

        self.actions_buf = actions.to(self.device)

        return (
            self.obs_buf,
            self.rew_buf,
            self.terminated_buf,
            self.truncated_buf,
            self.extras,
        )

    @abstractmethod
    def reset(self) -> ResetReturn:
        assert self._is_post_constructed, "Task not constructed: tried to reset"

        return self.obs_buf, self.extras

    @abstractmethod
    def get_observations(self) -> TaskObservations:
        return self.obs_buf, self.extras

    def render(self) -> Tuple[RenderFrame, ...] | None:
        pass
