from typing import Optional

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

import torch
from core.agents import NewtonBaseAgent
from core.animation import AnimationEngine
from core.envs import NewtonBaseEnv
from core.tasks import NewtonBaseTask, NewtonBaseTaskCallback
from core.types import Observations
from core.universe import Universe
from gymnasium.spaces import Box
from ..archiver.archiver import Config
from ..domain_randomizer import NewtonBaseDomainRandomizer
from ..domain_randomizer.domain_randomizer_old import DomainRandomizer


class NewtonIdleTaskCallback(NewtonBaseTaskCallback):
    def __init__(self, check_freq: int, save_path: str):
        super().__init__(check_freq, save_path)

        self.training_env: NewtonIdleTask

    def _on_step(self) -> bool:
        super()._on_step()

        # TODO: metrics about the agent's state & the animation engine

        return True


class NewtonIdleTask(NewtonBaseTask):
    def __init__(
        self,
        universe: Universe,
        env: NewtonBaseEnv,
        agent: NewtonBaseAgent,
        animation_engine: AnimationEngine,
        num_envs: int,
        device: str,
        playing: bool,
        reset_in_play: bool,
        max_episode_length: int,
        dr_configurations: Config,
    ):
        self.observation_space: Box = Box(
            low=np.array(
                [-10.0] * 3  # for the projected gravity
                + [-50.0] * 6  # for linear & angular velocities
                + [-1.0] * 12  # for the joint positions
                + [-1.0] * 12  # for the joint velocities
                + [-1.0] * 24  # for the previous 2 actions
                + [-1.0] * 2,  # for the transformed phase signal
            ),
            high=np.array(
                [10.0] * 3  # for the projected gravity
                + [50.0] * 6  # for linear & angular velocities
                + [1.0] * 12  # for the joint positions
                + [1.0] * 12  # for the joint velocities
                + [1.0] * 24  # for the previous 2 actions
                + [1.0] * 2  # for the transformed phase signal,
            ),
        )

        self.action_space: Box = Box(
            low=np.array([-1.0] * 12),
            high=np.array([1.0] * 12),
        )

        self.reward_space: Box = Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
        )

        super().__init__(
            universe,
            "newton_idle",
            env,
            agent,
            animation_engine,
            num_envs,
            device,
            playing,
            reset_in_play,
            max_episode_length,
            self.observation_space,
            self.action_space,
            self.reward_space,
            dr_configurations,
        )

        self.env: NewtonBaseEnv = env
        self.reset_height: float = 0.1

        if self.randomize:
            self.domain_randomizer: DomainRandomizer = DomainRandomizer(
                universe,
                self.num_envs,
                self.agent.base_path_expr,
                dr_configurations,
            )

    def construct(self) -> None:
        super().construct()

        self._is_constructed = True

    def post_construct(self):
        super().post_construct()

        self._is_post_constructed = True

    def step_wait(self) -> VecEnvStepReturn:
        super().step_wait()

        self.env.step(self.actions_buf)

        obs_buf = self._get_observations()
        self._update_rewards_and_dones()

        # only now can we save the actions (after gathering observations & rewards)
        self.last_actions_buf[1, :, :] = self.last_actions_buf[0, :, :]
        self.last_actions_buf[0, :, :] = self.actions_buf.clone()

        # creates a new np array with only the indices of the environments that are done
        resets: torch.Tensor = self.dones_buf.nonzero().squeeze(1)
        if len(resets) > 0:
            self.domain_randomizer.on_reset(resets)

        # clears the last 2 observations & the progress if any Newton is reset
        obs_buf[resets, :] = 0.0
        self.progress_buf[resets] = 0
        self.last_actions_buf[:, resets, :] = 0.0

        for i in range(self.num_envs):
            self.infos_buf[i] = {
                "TimeLimit.truncated": self.progress_buf[i] >= self.max_episode_length
            }

        return (
            obs_buf.cpu().numpy(),
            self.rewards_buf.cpu().numpy(),
            self.dones_buf.cpu().numpy(),
            self.infos_buf,
        )

    def reset(self) -> VecEnvObs:
        super().reset()

        obs_buf = self._get_observations()

        self.env.reset()

        # we want to return the last observation of the previous episode, according to the STB3 documentation
        return obs_buf.cpu().numpy()

    def _get_observations(self) -> Observations:
        obs = self.env.get_observations()

        phase_signal = self.progress_buf / self.max_episode_length

        obs_buf: Observations = torch.zeros(
            (self.num_envs, self.num_observations),
            dtype=torch.float32,
        )
        obs_buf[:, :3] = obs["projected_gravities"]
        obs_buf[:, 3:6] = obs["linear_velocities"]
        obs_buf[:, 6:9] = obs["angular_velocities"]
        obs_buf[:, 9:21] = self.agent.joints_controller.get_normalized_joint_positions()
        obs_buf[:, 21:33] = (
            self.agent.joints_controller.get_normalized_joint_velocities()
        )

        # 1st & 2nd set of past actions, we don't care about just-applied actions
        obs_buf[:, 33:57] = self.last_actions_buf.reshape(
            (self.num_envs, self.num_actions * 2)
        )

        # From what I currently understand, we add two observations to the observation space:
        #  - The phase signal (0-1) of the phase progress, transformed by cos(2*pi*progress) & sin(2*pi*progress)
        # This would allow the model to recognize the periodicity of the animation and hopefully learn the intricacies
        # of each gait.
        obs_buf[:, 57] = torch.cos(2 * torch.pi * phase_signal)
        obs_buf[:, 58] = torch.sin(2 * torch.pi * phase_signal)

        obs_buf = torch.clip(
            obs_buf,
            torch.from_numpy(self.observation_space.low).to(obs_buf.device),
            torch.from_numpy(self.observation_space.high).to(obs_buf.device),
        )

        return obs_buf

    def _update_rewards_and_dones(self) -> None:
        obs = self.env.get_observations()
        positions = obs["positions"]
        angular_velocities = obs["angular_velocities"]
        linear_velocities = obs["linear_velocities"]
        world_gravities = obs["world_gravities"]
        world_gravities_norm = world_gravities / torch.linalg.vector_norm(
            world_gravities,
            dim=1,
            keepdim=True,
        )
        projected_gravities = obs["projected_gravities"]
        projected_gravities_norm = projected_gravities / torch.linalg.vector_norm(
            projected_gravities,
            dim=1,
            keepdim=True,
        )
        in_contact_with_ground = obs["in_contacts"]

        dof_ordered_names = self.agent.joints_controller.art_view.dof_names
        has_flipped = projected_gravities_norm[:, 2] > 0.0

        self.air_time += torch.where(
            in_contact_with_ground,
            0.0,
            ~in_contact_with_ground * self._universe.control_dt,
        )

        terminated_by_long_airtime = torch.logical_and(
            # less than half a second of overall airtime (all paws)
            torch.sum(self.air_time, dim=1) > 5.0,
            # ensures that the agent has time to stabilize (0.5s)
            (self.progress_buf > 0.5 // self._universe.control_dt).to(self.device),
        )

        base_linear_velocity_xy = linear_velocities[:, :2]
        base_linear_velocity_z = linear_velocities[:, 2]
        base_angular_velocity_xy = angular_velocities[:, :2]
        base_angular_velocity_z = angular_velocities[:, 2]

        joint_positions = (
            self.agent.joints_controller.get_normalized_joint_positions()
        )  # [-1, 1] unitless
        joint_velocities = (
            self.agent.joints_controller.get_normalized_joint_velocities()
        )  # [-1, 1] unitless
        # joint_accelerations
        joint_efforts = (
            self.agent.joints_controller.get_normalized_joint_efforts()
        )  # [-1, 1] unitless

        animation_joint_data = self.animation_engine.get_multiple_clip_data_at_seconds(
            self.progress_buf * self._universe.control_dt,
            dof_ordered_names,
        )
        # we use the joint controller here, because it contains all the required information
        animation_joint_positions = (
            self.agent.joints_controller.normalize_joint_positions(
                animation_joint_data[:, :, 7]
            ).to(device=self.device)
        )  # [-1, 1] unitless
        animation_joint_velocities = (
            self.agent.joints_controller.normalize_joint_velocities(
                animation_joint_data[:, :, 8]
            ).to(device=self.device)
        )  # [-1, 1] unitless

        # DONES

        # terminated agents (i.e. they failed)
        terminated = torch.logical_or(has_flipped, terminated_by_long_airtime)

        # truncated agents (i.e. they reached the max episode length)
        truncated = (self.progress_buf >= self.max_episode_length).to(self.device)

        # when it's either terminated or truncated, the agent is done
        self.dones_buf = (
            torch.zeros_like(self.dones_buf)
            if not self.reset_in_play and self.playing
            else torch.logical_or(terminated, truncated)
        )

        # REWARDS

        from core.utils.rl import (
            squared_norm,
            exp_squared,
            exp_squared_norm,
            exp_squared_dot,
            fd_first_order_squared_norm,
            fd_second_order_squared_norm,
        )

        position_reward = exp_squared_norm(
            self.env.reset_newton_positions - positions,
            mult=-0.5,
            weight=0.5,
        )
        base_orientation_reward = exp_squared_dot(
            projected_gravities_norm,
            world_gravities_norm,
            mult=-20.0,
            weight=1.5,
        )
        base_linear_velocity_xy_reward = exp_squared_norm(
            base_linear_velocity_xy,
            mult=-8.0,
            weight=1.5,
        )
        base_linear_velocity_z_reward = exp_squared(
            base_linear_velocity_z,
            mult=-8.0,
            weight=1.0,
        )
        base_angular_velocity_xy_reward = exp_squared_norm(
            base_angular_velocity_xy,
            mult=-2,
            weight=0.5,
        )
        base_angular_velocity_z_reward = exp_squared(
            base_angular_velocity_z,
            mult=-2,
            weight=0.5,
        )
        joint_positions_reward = fd_first_order_squared_norm(
            joint_positions,
            animation_joint_positions,
            weight=15.0,
        )
        joint_velocities_reward = fd_first_order_squared_norm(
            joint_velocities,
            animation_joint_velocities,
            weight=0.01,
        )
        joint_efforts_reward = squared_norm(
            joint_efforts,
            weight=0.001,
        )
        # joint accelerations reward
        joint_action_rate_reward = fd_first_order_squared_norm(
            self.actions_buf,
            self.last_actions_buf[0],
            weight=1.0,
        )
        joint_action_acceleration_reward = fd_second_order_squared_norm(
            self.actions_buf,
            self.last_actions_buf[0],
            self.last_actions_buf[1],
            weight=0.45,
        )
        air_time_penalty = -torch.sum(self.air_time - 0.5, dim=1) * 2.0
        survival_reward = torch.where(
            terminated,
            0.0,
            2.0,
        )

        self.rewards_buf = (
            position_reward
            + base_orientation_reward
            + base_linear_velocity_xy_reward
            + base_linear_velocity_z_reward
            + base_angular_velocity_xy_reward
            + base_angular_velocity_z_reward
            + joint_positions_reward
            + joint_velocities_reward
            + joint_efforts_reward
            + joint_action_rate_reward
            + joint_action_acceleration_reward
            + air_time_penalty
            + survival_reward
        )

        self.rewards_buf *= self._universe.control_dt
