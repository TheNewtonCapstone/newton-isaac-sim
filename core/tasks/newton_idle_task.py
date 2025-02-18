from typing import Optional

import numpy as np

import torch
from core.agents import NewtonBaseAgent
from core.animation import AnimationEngine
from core.envs import NewtonBaseEnv
from core.tasks import NewtonBaseTask
from core.types import Observations, Actions, StepReturn, ResetReturn, Indices
from core.universe import Universe
from gymnasium.spaces import Box


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
    ):
        observation_space: Box = Box(
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

        action_space: Box = Box(
            low=np.array([-1.0] * 12),
            high=np.array([1.0] * 12),
        )

        reward_space: Box = Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
        )

        super().__init__(
            universe,
            "newton_idle",
            env,
            agent,
            animation_engine,
            None,
            num_envs,
            device,
            playing,
            reset_in_play,
            max_episode_length,
            observation_space,
            action_space,
            reward_space,
        )

        self.env: NewtonBaseEnv = env

    def construct(self) -> None:
        super().construct()

        self._is_constructed = True

    def post_construct(self):
        super().post_construct()

        self._is_post_constructed = True

    def step(self, actions: Actions) -> StepReturn:
        super().step(actions)

        self.env.step(self.actions_buf)

        self._update_observations_and_extras()
        self._update_rewards_and_dones()

        # only now can we save the actions (after gathering observations & rewards)
        self.last_actions_buf = self.actions_buf.clone()

        # creates a new np array with only the indices of the environments that are done
        resets: torch.Tensor = self.dones_buf.nonzero().squeeze(1)
        if len(resets) > 0:
            self.env.reset(resets)

        # clears the last 2 observations & the progress if any Newton is reset
        self.obs_buf[resets, :] = 0.0
        self.episode_length_buf[resets] = 0
        self.last_actions_buf[resets, :] = 0.0

        self.extras["time_outs"] = self.dones_buf.clone()

        return (
            self.obs_buf,
            self.rewards_buf.unsqueeze(-1),
            self.terminated_buf.unsqueeze(-1) | self.should_reset,
            self.truncated_buf.unsqueeze(-1) | self.should_reset,
            self.extras,
        )

    def reset(self, indices: Optional[Indices] = None) -> ResetReturn:
        super().reset()

        self.env.reset()

        self._update_observations_and_extras()

        # we want to return the last observation of the previous episode, according to the STB3 documentation
        return self.get_observations()

    def _update_observations_and_extras(self) -> None:
        env_obs = self.env.get_observations()

        phase_signal = self.episode_length_buf / self.max_episode_length

        self.obs_buf[:, :3] = env_obs["projected_gravities"]
        self.obs_buf[:, 3:6] = env_obs["linear_velocities"]
        self.obs_buf[:, 6:9] = env_obs["angular_velocities"]
        self.obs_buf[:, 9:21] = (
            self.agent.joints_controller.get_normalized_joint_positions()
        )
        self.obs_buf[:, 21:33] = (
            self.agent.joints_controller.get_normalized_joint_velocities()
        )

        # 1st & 2nd set of past actions, we don't care about just-applied actions
        self.obs_buf[:, 33:57] = self.last_actions_buf.reshape(
            (self.num_envs, self.num_actions * 2)
        )

        # From what I currently understand, we add two observations to the observation space:
        #  - The phase signal (0-1) of the phase progress, transformed by cos(2*pi*progress) & sin(2*pi*progress)
        # This would allow the model to recognize the periodicity of the animation and hopefully learn the intricacies
        # of each gait.
        self.obs_buf[:, 57] = torch.cos(2 * torch.pi * phase_signal)
        self.obs_buf[:, 58] = torch.sin(2 * torch.pi * phase_signal)

        self.obs_buf = torch.clip(
            self.obs_buf,
            torch.from_numpy(self.observation_space.low).to(self.obs_buf.device),
            torch.from_numpy(self.observation_space.high).to(self.obs_buf.device),
        )

    def _update_rewards_and_dones(self) -> None:
        env_obs = self.env.get_observations()

        positions = env_obs["positions"]
        angular_velocities = env_obs["angular_velocities"]
        linear_velocities = env_obs["linear_velocities"]
        world_gravities = env_obs["world_gravities"]

        world_gravities_norm = world_gravities / torch.linalg.vector_norm(
            world_gravities,
            dim=1,
            keepdim=True,
        )
        projected_gravities = env_obs["projected_gravities"]
        projected_gravities_norm = projected_gravities / torch.linalg.vector_norm(
            projected_gravities,
            dim=1,
            keepdim=True,
        )
        in_contact_with_ground = env_obs["in_contacts"]

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
            (self.episode_length_buf > 0.5 // self._universe.control_dt).to(
                self.device
            ),
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
            self.episode_length_buf * self._universe.control_dt,
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
        self.terminated_buf = has_flipped | terminated_by_long_airtime

        # truncated agents (i.e. they reached the max episode length)
        self.truncated_buf = (self.episode_length_buf >= self.max_episode_length).to(
            self.device
        )

        # REWARDS

        from core.utils.rl import (
            squared_norm,
            exp_squared,
            exp_squared_norm,
            exp_one_minus_squared_dot,
            fd_first_order_squared_norm,
        )

        position_reward = exp_squared_norm(
            self.env.reset_newton_positions - positions,
            mult=-0.5,
            weight=self.reward_scalers["position"],
        )
        base_orientation_reward = exp_one_minus_squared_dot(
            projected_gravities_norm,
            world_gravities_norm,
            mult=-20.0,
            weight=self.reward_scalers["base_orientation"],
        )
        base_linear_velocity_xy_reward = exp_squared_norm(
            base_linear_velocity_xy,
            mult=-8.0,
            weight=self.reward_scalers["base_linear_velocity_xy"],
        )
        base_linear_velocity_z_reward = exp_squared(
            base_linear_velocity_z,
            mult=-8.0,
            weight=self.reward_scalers["base_linear_velocity_z"],
        )
        base_angular_velocity_xy_reward = exp_squared_norm(
            base_angular_velocity_xy,
            mult=-2,
            weight=self.reward_scalers["base_angular_velocity_xy"],
        )
        base_angular_velocity_z_reward = exp_squared(
            base_angular_velocity_z,
            mult=-2,
            weight=self.reward_scalers["base_angular_velocity_z"],
        )
        joint_positions_reward = fd_first_order_squared_norm(
            joint_positions,
            animation_joint_positions,
            weight=-self.reward_scalers["joint_positions"],
        )
        joint_velocities_reward = fd_first_order_squared_norm(
            joint_velocities,
            animation_joint_velocities,
            weight=-self.reward_scalers["joint_velocities"],
        )
        joint_efforts_reward = squared_norm(
            joint_efforts,
            weight=-self.reward_scalers["joint_efforts"],
        )
        joint_action_rate_reward = fd_first_order_squared_norm(
            self.actions_buf,
            self.last_actions_buf,
            weight=-self.reward_scalers["joint_action_rate"],
        )
        joint_action_acceleration_reward = fd_first_order_squared_norm(
            self.actions_buf,
            self.last_actions_buf,
            weight=-self.reward_scalers["joint_action_acceleration"],
        )
        air_time_penalty = -torch.sum(self.air_time - 0.5, dim=1) * 2.0
        survival_reward = torch.where(
            self.dones_buf,
            0.0,
            self.reward_scalers["survival"],
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

        self.extras["episode"] = {
            "position_reward": position_reward.mean(),
            "base_orientation_reward": base_orientation_reward.mean(),
            "base_linear_velocity_xy_reward": base_linear_velocity_xy_reward.mean(),
            "base_linear_velocity_z_reward": base_linear_velocity_z_reward.mean(),
            "base_angular_velocity_z_reward": base_angular_velocity_z_reward.mean(),
            "joint_positions_reward": joint_positions_reward.mean(),
            "joint_action_rate_reward": joint_action_rate_reward.mean(),
            "joint_action_acceleration_reward": joint_action_acceleration_reward.mean(),
            "survival_reward": survival_reward.mean(),
            "terminated": self.terminated_buf.sum(),
            "truncated": self.truncated_buf.sum(),
        }
