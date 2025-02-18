from abc import abstractmethod
from typing import Optional

import torch as th
from gymnasium import Space
from gymnasium.spaces import Box

from .base_task import BaseTask
from ..agents import NewtonBaseAgent
from ..animation import AnimationEngine
from ..controllers import CommandController
from ..envs import NewtonBaseEnv
from ..types import (
    Actions,
    StepReturn,
    ResetReturn,
    ObservationScalers,
    ActionScaler,
    RewardScalers,
    TaskObservations,
)
from ..universe import Universe


class NewtonBaseTask(BaseTask):
    def __init__(
        self,
        universe: Universe,
        name: str,
        env: NewtonBaseEnv,
        agent: NewtonBaseAgent,
        animation_engine: Optional[AnimationEngine],
        command_controller: Optional[CommandController],
        num_envs: int,
        device: str,
        playing: bool,
        reset_in_play: bool,
        max_episode_length: int,
        observation_space: Space,
        action_space: Box,
        reward_space: Box,
        observation_scalers: Optional[ObservationScalers] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scalers: Optional[RewardScalers] = None,
    ):

        super().__init__(
            universe,
            name,
            env,
            agent,
            num_envs,
            device,
            playing,
            reset_in_play,
            max_episode_length,
            observation_space,
            action_space,
            reward_space,
            observation_scalers,
            action_scaler,
            reward_scalers,
        )

        self.training_env: NewtonBaseEnv = env
        self.agent: NewtonBaseAgent = agent

        self.animation_engine: Optional[AnimationEngine] = animation_engine
        self.command_controller: Optional[CommandController] = command_controller

        self.air_time: th.Tensor = th.zeros(
            (self.num_envs, 4),
            device=self.device,
        )  # air time per paw
        self.last_actions_buf: Actions = th.zeros(
            (self.num_envs, self.num_actions),
            dtype=th.float32,
            device=self.device,
        )

    @abstractmethod
    def construct(self) -> None:
        super().construct()

        self.env.register_self()

        if self.animation_engine:
            self.animation_engine.register_self(self.name)

        if self.command_controller:
            self.command_controller.register_self()

    def post_construct(self):
        super().post_construct()

    @abstractmethod
    def step(self, actions: Actions) -> StepReturn:
        self.episode_length_buf += 1

        return super().step(actions)

    @abstractmethod
    def reset(self) -> ResetReturn:
        return super().reset()

    def get_observations(self) -> TaskObservations:
        return self.obs_buf, self.extras
