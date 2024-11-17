from abc import ABC, abstractmethod

from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

import torch

from core.agents import BaseAgent, NewtonBaseAgent
from core.animation import AnimationEngine
from core.envs import BaseEnv
from core.envs.newton_base_env import NewtonBaseEnv
from core.tasks.base_task import BaseTask, BaseTaskCallback
from gymnasium import Space
from gymnasium.spaces import Box


class NewtonBaseTaskCallback(BaseTaskCallback):
    def __init__(self):
        super().__init__()

        self.training_env: NewtonBaseTask

    def _on_step(self) -> bool:
        super()._on_step()

        # TODO: metrics about the agent's state & the animation engine

        return True


class NewtonBaseTask(BaseTask):
    def __init__(
        self,
        training_env: BaseEnv,
        playing_env: BaseEnv,
        agent: BaseAgent,
        num_envs: int,
        device: str,
        headless: bool,
        playing: bool,
        max_episode_length: int,
        observation_space: Space,
        action_space: Box,
        reward_space: Box,
    ):
        self.training_env: NewtonBaseEnv
        self.playing_env: NewtonBaseEnv
        self.agent: NewtonBaseAgent

        self.animation_engine: AnimationEngine = AnimationEngine()  # TODO

        super().__init__(
            training_env,
            playing_env,
            agent,
            num_envs,
            device,
            headless,
            playing,
            max_episode_length,
            observation_space,
            action_space,
            reward_space,
        )

    @abstractmethod
    def construct(self) -> None:
        super().construct()

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        return super().step_wait()

    @abstractmethod
    def reset(self) -> VecEnvObs:
        return super().reset()
