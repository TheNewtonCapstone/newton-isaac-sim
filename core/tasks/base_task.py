from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Type

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvStepReturn,
    VecEnvObs,
)

import gymnasium
from core.agents import BaseAgent
from core.envs import BaseEnv
from core.types import Progress, Rewards, Dones, Actions, Infos
from core.universe import Universe


class BaseTaskCallback(BaseCallback):
    def _on_step(self) -> bool:
        task: BaseTask = self.training_env

        self.logger.record("rewards/median", torch.median(task.rewards_buf).item())

        self.logger.record("progress/mean", task.progress_buf.mean().item())

        self.logger.record("actions/mean", task.actions_buf.mean().item())

        return True


class BaseTask(VecEnv):
    def __init__(
        self,
        name: str,
        training_env: BaseEnv,
        playing_env: BaseEnv,
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
        self.name: str = name
        self.device: str = device
        self.playing: bool = playing

        self.agent: BaseAgent = agent
        self.env: BaseEnv = playing_env if self.playing else training_env

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

        super(BaseTask, self).__init__(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        self._is_constructed: bool = False

    @property
    def __str__(self):
        return f"BaseTask: {self.num_envs} environments, {self.num_observations} observations, {self.num_actions} actions"

    # TODO: assert that the task is (and is not, depending) constructed
    @abstractmethod
    def construct(self, universe: Universe) -> None:
        pass

    # Gymnasium methods (required from VecEnv)

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        pass

    @abstractmethod
    def reset(self) -> VecEnvObs:
        pass

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
