from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Type

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvStepReturn,
    VecEnvObs,
)

import gymnasium
from ..agents import BaseAgent
from ..base import BaseObject
from ..envs import BaseEnv
from ..types import Progress, Rewards, Dones, Actions, Infos
from ..universe import Universe


class BaseTaskCallback(BaseCallback):
    def _init_callback(self) -> None:
        task: BaseTask = self.training_env

        self.logger.record("meta/name", task.name)
        self.logger.record("meta/agent/name", task.agent.__class__.__name__)

        self.logger.record("meta/device", task.device)

        self.logger.record("meta/num_envs", task.num_envs)
        self.logger.record("meta/max_episode_length", task.max_episode_length)
        self.logger.record("meta/rl_step_dt", task.rl_step_dt)

        self.logger.record("meta/observation_space", task.observation_space)
        self.logger.record("meta/num_observations", task.num_observations)
        self.logger.record("meta/action_space", task.action_space)
        self.logger.record("meta/num_actions", task.num_actions)
        self.logger.record("meta/reward_space", task.reward_space)

    def _on_step(self) -> bool:
        task: BaseTask = self.training_env

        self.logger.record("rewards/median", torch.median(task.rewards_buf).item())

        self.logger.record("progress/mean", task.progress_buf.mean().item())

        self.logger.record("actions/mean", task.actions_buf.mean().item())

        return True


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
        max_episode_length: int,
        rl_step_dt: float,
        observation_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Box,
        reward_space: gymnasium.spaces.Box,
    ):
        BaseObject.__init__(
            self,
            universe=universe,
        )

        self.register_self()

        self.name: str = name
        self.device: str = device
        self.playing: bool = playing

        self.agent: BaseAgent = agent
        self.env: BaseEnv = env

        self.observation_space: gymnasium.spaces.Space = observation_space
        self.action_space: gymnasium.spaces.Box = action_space
        self.reward_space: gymnasium.spaces.Box = reward_space

        self.num_envs: int = num_envs
        self.max_episode_length: int = max_episode_length
        self.rl_step_dt: float = rl_step_dt

        self.num_observations: int = self.observation_space.shape[0]
        self.num_actions: int = self.action_space.shape[0]

        self.actions_buf: Actions = torch.zeros(
            (self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.rewards_buf: Rewards = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.dones_buf: Dones = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.progress_buf: Progress = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.infos_buf: Infos = [{} for _ in range(self.num_envs)]

        self.render_mode: str = "human"

        VecEnv.__init__(
            self,
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

    def __repr__(self):
        return f"BaseTask: {self.num_envs} environments, {self.num_observations} observations, {self.num_actions} actions"

    @abstractmethod
    def construct(self) -> None:
        BaseObject.construct(self)

    def post_construct(self):
        BaseObject.post_construct(self)

    # Gymnasium methods (required from VecEnv)

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        assert self._is_post_constructed, "Task not constructed: tried to step"

        return (
            np.zeros(self.num_envs),
            np.zeros(self.num_envs),
            np.zeros(self.num_envs),
            self.infos_buf,
        )

    @abstractmethod
    def reset(self) -> VecEnvObs:
        assert self._is_post_constructed, "Task not constructed: tried to reset"

        return {}

    def close(self) -> None:
        pass

    def seed(self, seed: Optional[int] = None) -> Sequence[None | int]:
        pass

    def step_async(self, actions: Actions) -> None:
        self.actions_buf = torch.from_numpy(actions).to(self.device)
        return

    # Helper methods (shouldn't need to be overridden and most likely won't be called directly)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        return [getattr(self, attr_name)]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Generally sets attribute inside vectorized environments (see base class). Unused."""
        pass

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        """Generally calls instance methods of vectorized environments. Unused."""
        return []

    def env_is_wrapped(
        self, wrapper_class: Type[gymnasium.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Generally checks if worker environments are wrapped with a given wrapper. Unused."""
        return [False]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gymnasium.Env]:
        """Generally gets the worker environments that should be targeted by given indices. Unused."""
        return [self]
