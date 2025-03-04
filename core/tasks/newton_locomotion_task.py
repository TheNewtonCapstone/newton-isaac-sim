import math
from typing import Optional

import numpy as np
import torch as th
from gymnasium.spaces import Box

from . import NewtonBaseTask
from ..agents import NewtonBaseAgent
from ..animation import AnimationEngine
from ..controllers import CommandController
from ..envs import NewtonBaseEnv
from ..types import (
    StepReturn,
    TaskObservations,
    ResetReturn,
    ObservationScalers,
    ActionScaler,
    RewardScalers,
    CommandScalers,
    Indices,
)
from ..universe import Universe


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
        observation_scalers: Optional[ObservationScalers] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scalers: Optional[RewardScalers] = None,
        command_scalers: Optional[CommandScalers] = None,
    ):
        observation_space: Box = Box(
            low=np.array(
                [-np.Inf] * 47,
                dtype=np.float32,
            ),
            high=np.array(
                [np.Inf] * 47,
                dtype=np.float32,
            ),
        )

        action_space: Box = Box(
            low=np.array([-1.0] * 12),
            high=np.array([1.0] * 12),
            dtype=np.float32,
        )

        reward_space: Box = Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32,
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
            observation_space,
            action_space,
            reward_space,
            observation_scalers,
            action_scaler,
            reward_scalers,
        )

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

        self.command_scalers: Optional[CommandScalers] = command_scalers

    def construct(self) -> None:
        super().construct()

        self._is_constructed = True

    def post_construct(self):
        super().post_construct()

        self._is_post_constructed = True

    def step(self, actions) -> StepReturn:
        super().step(actions * self._action_scaler)

        self.env.step(self._actions_buf)

        self._update_observations_and_extras()
        self._update_rewards_and_dones()

        # only now can we save the actions (after gathering observations & rewards)
        self.last_actions_buf = self._actions_buf.clone()

        # re-compute where the agents are supposed to be, based on the current command velocities
        self.predicted_base_positions_xy += (
            self.current_velocity_commands_xy * self._universe.control_dt
        )

        # creates a new np array with only the indices of the environments that are done
        resets: th.Tensor = (self.dones_buf & self.should_reset).nonzero().squeeze(1)
        if len(resets) > 0:
            self._update_terrain_curriculumn(resets)
            self.env.reset(resets)

        # clears the last 2 observations, the progress & the predicted positions if any Newton is reset
        self.obs_buf[resets, :] = 0.0
        self._episode_length_buf[resets] = 0
        self.last_actions_buf[resets, :] = 0.0
        self.predicted_base_positions_xy[resets, :] = self.env.reset_newton_positions[
            resets, :2
        ]

        self._extras["time_outs"] = self.dones_buf.clone()

        command_resets = (self._episode_length_buf % 250 == 0).nonzero().squeeze(1)
        self._update_velocity_commands(command_resets)

        return (
            self.obs_buf,
            self.rew_buf.unsqueeze(-1),
            self.terminated_buf.unsqueeze(-1) & self.should_reset,
            self.truncated_buf.unsqueeze(-1) & self.should_reset,
            self._extras,
        )

    def reset(self, indices: Optional[Indices] = None) -> ResetReturn:
        super().reset()

        self.env.reset()

        # we're starting anew, new velocities for all
        self._update_velocity_commands()
        self._update_observations_and_extras()

        return self.get_observations()

    def _update_observations_and_extras(self) -> None:
        env_obs = self.env.get_observations()

        self._obs_buf[:, :3] = env_obs["projected_gravities"]

        # we normalize the linear & angular velocities to [-1, 1]
        self._obs_buf[:, 3:6] = env_obs["linear_velocities"]
        self._obs_buf[:, 6:9] = env_obs["angular_velocities"]

        self._obs_buf[:, 9:21] = self.agent.joints_controller.get_joint_positions_rad()
        self._obs_buf[:, 21:33] = (
            self.agent.joints_controller.get_joint_velocities_rad()
        )

        # 1st & 2nd set of past actions, we don't care about just-applied actions
        self._obs_buf[:, 33:45] = self.last_actions_buf.clone()

        self._obs_buf[:, 45:47] = self.current_velocity_commands_xy

        self._obs_buf = th.clip(
            self._obs_buf,
            th.from_numpy(self.observation_space.low).to(self.device),
            th.from_numpy(self.observation_space.high).to(self.device),
        )

        self._extras["time_outs"] = self.dones_buf.clone()

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

        has_flipped = projected_gravities_norm[:, 2] > 0.0

        # based on the projected gravity, we can determine if Newton
        # is tilted by more than x degrees
        is_tilted = th.acos(
            th.clamp(
                th.sum(
                    world_gravities_norm * projected_gravities_norm,
                    dim=1,
                ),
                -1.0,
                1.0,
            )
        ) > math.radians(20)

        self.air_time = th.where(
            in_contact_with_ground,
            0.0,
            self.air_time + ~in_contact_with_ground * self._universe.control_dt,
        )

        terminated_by_long_airtime = th.logical_and(
            # less than half a second of overall airtime (all paws)
            th.sum(self.air_time, dim=1) > 8.0,
            # ensures that the agent has time to stabilize (0.5s)
            (self._episode_length_buf > 0.5 // self._universe.control_dt).to(
                self.device
            ),
        )

        base_positions_xy = positions[:, :2]
        base_position_z = positions[:, 2]

        base_linear_velocity_xy = linear_velocities[:, :2]
        base_linear_velocity_z = linear_velocities[:, 2]
        base_angular_velocity_xy = angular_velocities[:, :2]
        base_angular_velocity_z = angular_velocities[:, 2]

        joint_positions = (
            self.agent.joints_controller.get_joint_positions_deg()
        )  # [-1, 1] unitless

        dof_ordered_names = self.agent.joints_controller.art_view.dof_names
        animation_joint_data = self.animation_engine.get_multiple_clip_data_at_seconds(
            self._episode_length_buf * self._universe.control_dt,
            dof_ordered_names,
        )
        # we use the joint controller here, because it contains all the required information
        animation_joint_positions = (
            # self.agent.joints_controller.normalize_joint_positions(
            animation_joint_data[:, :, 7].to(self.device)
            # ).to(device=self.device)
        )  # [-1, 1] unitless

        # DONES

        # terminated agents (i.e. they failed)
        self._terminated_buf = (
            has_flipped  # has_flipped | is_tilted | terminated_by_long_airtime
        )

        # truncated agents (i.e. they reached the max episode length)
        self._truncated_buf = (self._episode_length_buf >= self._max_episode_length).to(
            self.device
        )

        # REWARDS

        from core.utils.rl.rewards import (
            squared,
            exp_squared,
            exp_one_minus_squared_dot,
            exp_fd_first_order_squared_norm,
            fd_first_order_squared_norm,
            fd_first_order_squared,
            fd_first_order_sum_abs,
        )

        position_reward = exp_fd_first_order_squared_norm(
            self.predicted_base_positions_xy,
            base_positions_xy,
            mult=-2.0,
            weight=self._reward_scalers["position"],
        )
        height_reward = fd_first_order_squared(
            self.env.reset_newton_positions[:, 2],
            base_position_z,
            weight=-self._reward_scalers["height"],
        )
        base_stability_reward = exp_one_minus_squared_dot(
            projected_gravities_norm,
            world_gravities_norm,
            mult=-5.0,
            weight=self._reward_scalers["base_stability"],
        )
        base_linear_velocity_xy_reward = exp_fd_first_order_squared_norm(
            self.current_velocity_commands_xy,
            base_linear_velocity_xy,
            mult=-4.0,
            weight=self._reward_scalers["base_linear_velocity_xy"],
        )
        base_linear_velocity_z_reward = squared(
            base_linear_velocity_z,
            weight=-self._reward_scalers["base_linear_velocity_z"],
        )
        base_angular_velocity_z_reward = exp_squared(
            base_angular_velocity_z,
            mult=-4.0,
            weight=self._reward_scalers["base_angular_velocity_z"],
        )
        joint_positions_reward = fd_first_order_sum_abs(
            joint_positions,
            animation_joint_positions,
            weight=-self._reward_scalers["joint_positions"],
        )
        joint_action_rate_reward = fd_first_order_squared_norm(
            self._actions_buf,
            self.last_actions_buf,
            weight=-self._reward_scalers["joint_action_rate"],
        )
        air_time_reward = squared(
            th.sum(self.air_time, dim=1),
            weight=-self._reward_scalers["air_time"],
        )
        survival_reward = th.where(
            self.terminated_buf,
            0.0,
            self._reward_scalers["survival"],
        )

        self._rew_buf = (
            # position_reward
            height_reward
            # + base_stability_reward
            + base_linear_velocity_xy_reward
            + base_linear_velocity_z_reward
            + base_angular_velocity_z_reward
            + joint_positions_reward
            + joint_action_rate_reward
            # + air_time_reward
            # + survival_reward
        )

        # self.rew_buf *= self._universe.control_dt

        self._extras["episode"] = {
            "position_reward": position_reward.mean(),
            "height_reward": height_reward.mean(),
            "base_stability_reward": base_stability_reward.mean(),
            "base_linear_velocity_xy_reward": base_linear_velocity_xy_reward.mean(),
            "base_linear_velocity_z_reward": base_linear_velocity_z_reward.mean(),
            "base_angular_velocity_z_reward": base_angular_velocity_z_reward.mean(),
            "joint_positions_reward": joint_positions_reward.mean(),
            "joint_action_rate_reward": joint_action_rate_reward.mean(),
            "air_time_reward": air_time_reward.mean(),
            "survival_reward": survival_reward.mean(),
            "terminated": self.terminated_buf.sum(),
            "truncated": self.truncated_buf.sum(),
        }

    def _update_velocity_commands(self, indices: Optional[th.Tensor] = None) -> None:
        if indices is None:
            indices = th.arange(self.num_envs, device=self.device)

        for i in indices:
            self.current_velocity_commands_xy[i, :2] = (
                self.command_controller.get_random_action()
            ) * self.command_scalers["linear_velocity_xy"]

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

        # The level is updated based on the distance traversed by the agent
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
