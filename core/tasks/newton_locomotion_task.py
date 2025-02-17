from typing import Optional

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

import torch as th
from core.agents import NewtonBaseAgent
from core.animation import AnimationEngine
from core.controllers import CommandController
from core.envs import NewtonBaseEnv
from core.tasks import NewtonBaseTask, NewtonBaseTaskCallback
from core.types import Observations, Indices
from core.universe import Universe
from gymnasium.spaces import Box


class NewtonLocomotionTaskCallback(NewtonBaseTaskCallback):
    def __init__(self, check_freq: int, save_path: str):
        super().__init__(check_freq, save_path)

        self.training_env: NewtonLocomotionTask

    def _on_step(self) -> bool:
        super()._on_step()

        task: NewtonLocomotionTask = self.training_env

        self.logger.record(
            "observations/velocity_commands",
            task.current_velocity_commands_xy.norm(dim=1).median().item(),
        )

        return True


class NewtonLocomotionTask(NewtonBaseTask):
    def __init__(
        self,
        universe: Universe,
        env: NewtonBaseEnv,
        agent: NewtonBaseAgent,
        animation_engine: AnimationEngine,
        command_controller: CommandController,
        num_envs: int,
        device: str,
        playing: bool,
        reset_in_play: bool,
        max_episode_length: int,
    ):
        self.observation_space: Box = Box(
            low=np.array(
                [-10.0] * 3  # for the projected gravity
                + [-50.0] * 6  # for linear & angular velocities
                + [-1.0] * 12  # for the joint positions
                + [-1.0] * 12  # for the joint velocities
                + [-1.0] * 24  # for the previous 2 actions
                + [-1.0] * 2  # for the transformed phase signal
                + [-1.0] * 2,  # for the velocity command
            ),
            high=np.array(
                [10.0] * 3  # for the projected gravity
                + [50.0] * 6  # for linear & angular velocities
                + [1.0] * 12  # for the joint positions
                + [1.0] * 12  # for the joint velocities
                + [1.0] * 24  # for the previous 2 actions
                + [1.0] * 2  # for the transformed phase signal
                + [1.0] * 2,  # for the velocity command
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
            "newton_locomotion",
            env,
            agent,
            animation_engine,
            command_controller,
            num_envs,
            device,
            playing,
            reset_in_play,
            max_episode_length,
            self.observation_space,
            self.action_space,
            self.reward_space,
        )

        self.env: NewtonBaseEnv = env

        self.current_velocity_commands_xy: th.Tensor = th.zeros(
            (self.num_envs, 2),
            dtype=th.float32,
            device=self.device,
        )

        self.predicted_base_positions_xy = th.zeros(
            (self.num_envs, 2),
            dtype=th.float32,
            device=self.device,
        )

        self.curriculum_levels = th.zeros(num_envs, dtype=th.int16, device=self.device)

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

        # re-compute where the agents are supposed to be, based on the current command velocities
        self.predicted_base_positions_xy += (
            self.current_velocity_commands_xy * self._universe.control_dt
        )

        # creates a new np array with only the indices of the environments that are done
        resets: th.Tensor = self.dones_buf.nonzero().squeeze(1)
        if len(resets) > 0:
            self._update_terrain_curriculumn(resets)
            self.env.reset(resets)

        # clears the last 2 observations, the progress & the predicted positions if any Newton is reset
        obs_buf[resets, :] = 0.0
        self.progress_buf[resets] = 0
        self.last_actions_buf[:, resets, :] = 0.0

        self._update_velocity_commands(resets)
        self.predicted_base_positions_xy[resets, :] = self.env.reset_newton_positions[
            resets, :2
        ]

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

    def reset(self, indices: Indices = None) -> VecEnvObs:
        super().reset()

        obs_buf = self._get_observations()

        if self.env.terrain.curriculum:
            self._update_terrain_curriculumn(indices)

        self.env.reset()

        # we're starting anew, new velocities for all
        self._update_velocity_commands()

        # we want to return the last observation of the previous episode, according to the STB3 documentation
        return obs_buf.cpu().numpy()

    def _get_observations(self) -> Observations:
        obs = self.env.get_observations()

        phase_signal = self.progress_buf / self.max_episode_length

        obs_buf: Observations = th.zeros(
            (self.num_envs, self.num_observations),
            dtype=th.float32,
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
        obs_buf[:, 57] = th.cos(2 * th.pi * phase_signal)
        obs_buf[:, 58] = th.sin(2 * th.pi * phase_signal)

        obs_buf[:, 59:61] = self.current_velocity_commands_xy

        obs_buf = th.clip(
            obs_buf,
            th.from_numpy(self.observation_space.low).to(obs_buf.device),
            th.from_numpy(self.observation_space.high).to(obs_buf.device),
        )

        return obs_buf

    def _update_rewards_and_dones(self) -> None:
        obs = self.env.get_observations()
        positions = obs["positions"]
        angular_velocities = obs["angular_velocities"]
        linear_velocities = obs["linear_velocities"]
        world_gravities = obs["world_gravities"]
        world_gravities_norm = world_gravities / world_gravities.norm(
            dim=1,
            keepdim=True,
        )
        projected_gravities = obs["projected_gravities"]
        projected_gravities_norm = projected_gravities / projected_gravities.norm(
            dim=1,
            keepdim=True,
        )
        in_contact_with_ground = obs["in_contacts"]

        # hasn't move much from its original position in the last 0.5s
        # is_stagnant =
        has_flipped = projected_gravities_norm[:, 2] > 0.0

        self.air_time = th.where(
            in_contact_with_ground,
            0.0,
            self.air_time + ~in_contact_with_ground * self._universe.control_dt,
        )

        terminated_by_long_airtime = th.logical_and(
            # less than half a second of overall airtime (all paws)
            th.sum(self.air_time, dim=1) > 5.0,
            # ensures that the agent has time to stabilize (0.5s)
            (self.progress_buf > 0.5 // self._universe.control_dt).to(self.device),
        )

        base_positions_xy = positions[:, :2]

        base_linear_velocity_xy = linear_velocities[:, :2]
        base_linear_velocity_z = linear_velocities[:, 2]
        base_angular_velocity_xy = angular_velocities[:, :2]
        base_angular_velocity_z = angular_velocities[:, 2]

        joint_positions = (
            self.agent.joints_controller.get_normalized_joint_positions()
        )  # [-1, 1] unitless

        dof_ordered_names = self.agent.joints_controller.art_view.dof_names
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

        # DONES

        # terminated agents (i.e. they failed)
        terminated = th.logical_or(has_flipped, terminated_by_long_airtime)

        # truncated agents (i.e. they reached the max episode length)
        truncated = (self.progress_buf >= self.max_episode_length).to(self.device)

        # when it's either terminated or truncated, the agent is done
        self.dones_buf = (
            th.zeros_like(self.dones_buf)
            if not self.reset_in_play and self.playing
            else th.logical_or(terminated, truncated)
        )

        # REWARDS

        from core.utils.rl import (
            squared_norm,
            exp_squared,
            exp_squared_norm,
            exp_squared_dot,
            exp_fd_first_order_squared_norm,
            fd_first_order_squared_norm,
            fd_second_order_squared_norm,
        )

        position_reward = exp_fd_first_order_squared_norm(
            self.predicted_base_positions_xy,
            base_positions_xy,
            mult=-6.0,
            weight=1.5,
        )
        base_orientation_reward = exp_squared_dot(
            projected_gravities_norm,
            world_gravities_norm,
            mult=-5.0,
            weight=1.5,
        )
        base_linear_velocity_xy_reward = exp_fd_first_order_squared_norm(
            self.current_velocity_commands_xy,
            base_linear_velocity_xy,
            mult=-8.0,
            weight=2.0,
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
            weight=5.0,
        )
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
        air_time_penalty = -th.sum(self.air_time, dim=1) * 1.0
        survival_reward = th.where(
            terminated,
            0.0,
            10.0,
        )

        self.rewards_buf = (
            position_reward
            + base_orientation_reward
            + base_linear_velocity_xy_reward
            + base_linear_velocity_z_reward
            + base_angular_velocity_xy_reward
            + base_angular_velocity_z_reward
            + joint_positions_reward
            + joint_action_rate_reward
            + joint_action_acceleration_reward
            + air_time_penalty
            + survival_reward
        )

        self.rewards_buf *= self._universe.control_dt

    def _update_velocity_commands(self, indices: Optional[th.Tensor] = None) -> None:
        if indices is None:
            indices = th.arange(self.num_envs, device=self.device)

        for i in indices:
            self.current_velocity_commands_xy[i, 0:2] = (
                self.command_controller.get_random_action()
            )

    def _update_terrain_curriculumn(self, indices: Optional[th.Tensor] = None) -> None:
        if indices is None:
            return

        obs = self.env.get_observations()
        agent_heights = self.agent.transformed_position[2]
        flat_origins = th.tensor(
            self.env.terrain.sub_terrain_origins,
            dtype=th.float32,
            device=self.device,
        )
        flat_origins[:, 2] += agent_heights
        sub_terrain_length = self.env.terrain.sub_terrain_length

        level_indices = self.curriculum_levels[indices].long()

        # The levl is updated based on the distance traversed by the agent
        distance = obs["positions"][indices, :2] - flat_origins[level_indices, :2]
        distance = th.norm(distance, dim=1)
        move_up = distance >= sub_terrain_length / 2
        move_down = distance < sub_terrain_length / 2

        # Update the Newton levels
        self.curriculum_levels[indices] += 1 * move_up - 1 * move_down

        # Ensure levels stay within bounds
        max_level = self.env.terrain.num_sub_terrains - 1  # Max valid sub-terrain index
        self.curriculum_levels[indices] = th.clamp(
            self.curriculum_levels[indices],
            min=0,
            max=max_level,
        )

        # Ensure newton_levels is a valid index type
        level_indices = self.curriculum_levels[indices].long()

        # Get new spawn positions based on the levels
        new_spawn_positions = flat_origins[level_indices, :]

        # Update the initial positions in the environment
        self.env.domain_randomizer.set_initial_position(indices, new_spawn_positions)
