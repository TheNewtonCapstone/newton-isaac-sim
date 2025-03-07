from abc import abstractmethod
from typing import Optional, Tuple

import gymnasium
import torch
from gymnasium.core import RenderFrame

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


class BaseTask(BaseObject):
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
        state_space: Optional[gymnasium.spaces.Box] = None,
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

        self._name: str = name
        self._device: str = device
        self._playing: bool = playing
        self._reset_in_play: bool = reset_in_play

        self._agent: BaseAgent = agent
        self._env: BaseEnv = env

        self._observation_space: gymnasium.spaces.Space = observation_space
        self._state_space: Optional[gymnasium.spaces.Box] = state_space
        self._action_space: gymnasium.spaces.Box = action_space
        self._reward_space: gymnasium.spaces.Box = reward_space

        self._observation_scalers: Optional[ObservationScalers] = observation_scalers
        self._action_scaler: Optional[ActionScaler] = action_scaler
        self._reward_scalers: Optional[RewardScalers] = reward_scalers

        self._num_envs: int = num_envs
        self._max_episode_length: int = max_episode_length

        self._num_obs: int = self.observation_space.shape[0]
        self._num_actions: int = self.action_space.shape[0]

        self._obs_buf: Observations = torch.zeros(
            (self.num_envs, self._num_obs),
            dtype=torch.float32,
            device=self.device,
        )
        self._actions_buf: Actions = torch.zeros(
            (self.num_envs, self._num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self._rew_buf: Rewards = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._terminated_buf: Terminated = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._truncated_buf: Truncated = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._episode_length_buf: EpisodeLength = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._extras: Extras = {"episode": {}, "time_outs": torch.zeros(self.num_envs)}

    def __repr__(self):
        return f"BaseTask: {self.num_envs} environments, {self._num_obs} observations, {self._num_actions} actions"

    @property
    def name(self) -> str:
        return self._name

    @property
    def env(self) -> BaseEnv:
        return self._env

    @property
    def agent(self) -> BaseAgent:
        return self._agent

    @property
    def obs_buf(self) -> Observations:
        return self._obs_buf

    @property
    def actions_buf(self) -> Actions:
        return self._actions_buf

    @property
    def rew_buf(self) -> Rewards:
        return self._rew_buf

    @property
    def terminated_buf(self) -> Terminated:
        return self._terminated_buf

    @property
    def truncated_buf(self) -> Truncated:
        return self._truncated_buf

    @property
    def dones_buf(self) -> Dones:
        return self._terminated_buf | self._truncated_buf

    @property
    def episode_length_buf(self) -> EpisodeLength:
        return self._episode_length_buf

    @property
    def extras(self) -> Extras:
        return self._extras

    @property
    def should_reset(self) -> bool:
        return self._reset_in_play or not self._playing

    @property
    def observation_space(self) -> gymnasium.Space:
        return self._observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self._action_space

    @property
    def state_space(self) -> gymnasium.Space | None:
        return self._state_space

    @property
    def device(self) -> str:
        return self._device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_agents(self) -> int:
        return 1

    @abstractmethod
    def pre_build(self) -> None:
        super().pre_build()

    @abstractmethod
    def post_build(self):
        super().post_build()

    @abstractmethod
    def step(self, actions: Actions) -> StepReturn:
        assert self.is_built, f"{self.__class__.__name__} not built: tried to step"

        self._actions_buf = actions.to(self.device)

        return (
            self._obs_buf,
            self._rew_buf,
            self._terminated_buf,
            self._truncated_buf,
            self._extras,
        )

    @abstractmethod
    def reset(self) -> ResetReturn:
        assert self.is_built, f"{self.__class__.__name__} not built: tried to reset"

        return self._obs_buf, self._extras

    @abstractmethod
    def get_observations(self) -> TaskObservations:
        return self._obs_buf, self._extras

    def state(self) -> Observations:
        pass

    # generally unused by us, necessary for SKRL

    def close(self) -> None:
        pass

    def render(self) -> Tuple[RenderFrame, ...] | None:
        pass
