from typing import Dict, Any, Callable, List, Optional, Sequence, Type

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvStepReturn,
    VecEnvObs,
)

import gymnasium
import numpy as np
import torch
from core.base.base_agent import BaseAgent
from core.base.base_env import BaseEnv


class BaseTask(VecEnv):
    def __init__(
        self,
        headless: bool,
        device: str,
        num_envs: int,
        playing: bool,
        max_episode_length: int,
        observation_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Box,
        reward_space: gymnasium.spaces.Box,
        training_env_factory: Callable[..., BaseEnv],
        playing_env_factory: Callable[..., BaseEnv],
        agent_factory: Callable[..., BaseAgent],
    ):
        self.render_mode = "human"

        self.training_env_factory = training_env_factory
        self.playing_env_factory = playing_env_factory
        self.agent_factory = agent_factory

        self.agent: BaseAgent | None = None
        self.env: BaseEnv | None = None  # TODO: is this necessary?

        self.headless: bool = headless
        self.device: str = device
        self.playing: bool = playing

        self.observation_space: gymnasium.spaces.Space = observation_space
        self.action_space: gymnasium.spaces.Box = action_space
        self.reward_space: gymnasium.spaces.Box = reward_space

        self.num_envs: int = num_envs
        self.max_episode_length: int = max_episode_length

        self.num_observations: int = self.observation_space.shape[0]
        self.num_actions: int = self.action_space.shape[0]

        self.actions_buf: torch.Tensor = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float32
        )
        self.rewards_buf: torch.Tensor = torch.zeros(self.num_envs, dtype=torch.float32)
        self.dones_buf: torch.Tensor = torch.zeros(self.num_envs, dtype=torch.bool)
        self.progress_buf: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.float32
        )
        self.infos_buf: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

        super(BaseTask, self).__init__(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

    def construct(self) -> bool:
        pass

    def __str__(self):
        return f"{self.__class__.__name__} with {self.num_envs} environments, {self.num_observations} observations, {self.num_actions} actions, {self.num_states} states."

    # Gymnasium methods (required from VecEnv)

    def reset(self) -> VecEnvObs:
        pass

    def step_async(self, actions: np.ndarray) -> None:
        self.actions_buf = torch.from_numpy(actions)
        return

    def step_wait(self) -> VecEnvStepReturn:
        pass

    def close(self) -> None:
        pass

    def seed(self, seed: Optional[int] = None) -> Sequence[None | int]:
        pass

    # Helper methods (shouldn't need to be overridden)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        print(f"Calling {method_name} on {len(target_envs)} environments")
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(
        self, wrapper_class: Type[gymnasium.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gymnasium.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
