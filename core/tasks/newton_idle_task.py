from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

import numpy as np
from core.agents import NewtonBaseAgent
from core.animation import AnimationEngine
from core.envs import NewtonBaseEnv
from core.tasks import NewtonBaseTask, NewtonBaseTaskCallback
from core.universe import Universe
from gymnasium.spaces import Box


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
        training_env: NewtonBaseEnv,
        playing_env: NewtonBaseEnv,
        agent: NewtonBaseAgent,
        animation_engine: AnimationEngine,
        num_envs: int,
        device: str,
        playing: bool,
        max_episode_length: int,
    ):
        agent_joints_low: np.ndarray = agent.joints_controller.joint_constraints.low
        agent_joints_high: np.ndarray = agent.joints_controller.joint_constraints.high

        self.observation_space: Box = Box(
            low=np.array(
                [-1.0] * 3
                + [-np.Inf] * 7
                + agent_joints_low.tolist()  # twice because we give past actions (target positions) + current positions
                + agent_joints_low.tolist()
                + [-np.Inf] * 12
                + [-1] * 2,  # for the transformed phase signal
            ),
            high=np.array(
                [1.0] * 3
                + [np.Inf] * 7
                + agent_joints_high.tolist()
                + agent_joints_high.tolist()
                + [np.Inf] * 12
                + [1] * 2,
            ),
        )

        self.action_space: Box = Box(
            low=agent_joints_low.copy(),
            high=agent_joints_high.copy(),
        )

        self.reward_space: Box = Box(
            low=np.array([-np.Inf]),
            high=np.array([np.Inf]),
        )

        super().__init__(
            "newton_idle",
            training_env,
            playing_env,
            agent,
            animation_engine,
            num_envs,
            device,
            playing,
            max_episode_length,
            self.observation_space,
            self.action_space,
            self.reward_space,
        )

        self.reset_height: float = 0.1

    def construct(self, universe: Universe) -> None:
        super().construct(universe)

    def step_wait(self) -> VecEnvStepReturn:
        super().step_wait()

        self.env.step(self.actions_buf)

        # TODO: Integrate the animation engine into the Newton*Tasks
        #   Update the observations & reward function accordingly
        #   From what I currently understand, we add two observations to the observation space:
        #    - The phase signal (0-1) of the phase progress, transformed by cos(2*pi*progress) & sin(2*pi*progress)
        #   This would allow the model to recognize the periodicity of the animation and hopefully learn the intricacies
        #   of each gait.
        #   The reward function would compare desired joint positions with the current joint positions, and reward
        #   the agent for getting closer to the desired positions.

        phase_signal = self.progress_buf / self.max_episode_length

        obs = self._get_observations()
        heights = obs["positions"][:, 2]

        obs_buf: np.ndarray = np.zeros(
            (self.num_envs, self.num_observations), dtype=np.float32
        )
        obs_buf[:, :3] = obs["projected_gravities"]
        obs_buf[:, 3:6] = obs["linear_accelerations"]
        obs_buf[:, 6:9] = obs["angular_velocities"]
        obs_buf[:, 9] = heights
        obs_buf[:, 10:22] = self.actions_buf
        obs_buf[:, 22:34] = (
            self.agent.joints_controller.art_view.get_joint_positions().cpu().numpy()
        )
        obs_buf[:, 34:46] = (
            self.agent.joints_controller.art_view.get_joint_velocities().cpu().numpy()
        )

        # For the explanation of the phase signal, see the comment above
        obs_buf[:, 46] = np.cos(2 * np.pi * phase_signal)
        obs_buf[:, 47] = np.sin(2 * np.pi * phase_signal)

        self._calculate_rewards()

        self.last_actions_buf = self.actions_buf.copy()

        # terminated
        self.dones_buf = np.where(heights <= self.reset_height, True, False)
        # self.dones_buf = np.where(heights >= 1.0, True, self.dones_buf)

        self.dones_buf = np.where(
            self.progress_buf >= self.max_episode_length,
            True,
            self.dones_buf,
        )  # truncated

        # creates a new np array with only the indices of the environments that are done
        resets = self.dones_buf.nonzero()[0].flatten()
        if len(resets) > 0:
            self.env.reset(resets)

        # clears the last 2 observations & the progress if any Newton is reset
        obs_buf[resets, :] = 0.0
        self.progress_buf[resets] = 0
        self.last_actions_buf[resets, :] = 0.0

        return (
            obs_buf.copy(),
            self.rewards_buf.copy(),
            self.dones_buf.copy(),
            self.infos_buf,
        )

    def reset(self) -> VecEnvObs:
        super().reset()

        obs = self._get_observations()

        obs_buf: np.ndarray = np.zeros(
            (self.num_envs, self.num_observations), dtype=np.float32
        )

        obs_buf[:, :3] = obs["projected_gravities"]
        obs_buf[:, 3:6] = obs["linear_accelerations"]
        obs_buf[:, 6:9] = obs["angular_velocities"]
        obs_buf[:, 9] = obs["positions"][:, 2]
        obs_buf[:, 10:22] = self.actions_buf
        obs_buf[:, 22:34] = (
            self.agent.joints_controller.art_view.get_joint_positions().cpu().numpy()
        )
        obs_buf[:, 34:46] = (
            self.agent.joints_controller.art_view.get_joint_velocities().cpu().numpy()
        )

        self.env.reset()

        # we want to return the last observation of the previous episode
        return obs_buf

    def _get_observations(self) -> VecEnvObs:
        # TODO: make this function return an observation np.ndarray instead of a dict
        env_observations = self.env.get_observations()

        return env_observations

    def _calculate_rewards(self) -> None:
        # TODO: rework rewards for Newton*Tasks

        phase_signal = self.progress_buf / self.max_episode_length

        obs = self._get_observations()
        positions = obs["positions"]
        angular_velocities = obs["angular_velocities"]
        linear_velocities = obs["linear_velocities"]

        joints_order = self.agent.joints_controller.art_view.joint_names

        # TODO: Decide on whether to have an action space from [0, 1] or within the ROMs
        #   Disney uses an action space from [0, 1] along with some transformations to eventually map it to their motors ROMs
        #   Or we can keep doing what we're doing, but adjust the reward function to avoid huge numbers

        # base position
        # base orientation
        base_linear_velocity_xy = linear_velocities[:, :2]
        base_linear_velocity_z = linear_velocities[:, 2]
        base_angular_velocity_xy = angular_velocities[:, :2]
        base_angular_velocity_z = angular_velocities[:, 2]
        joint_angles = (
            np.rad2deg(
                self.agent.joints_controller.art_view.get_joint_positions()
                .cpu()
                .numpy()
            )
            / self.reward_space.high
        )  # in degrees
        joint_velocities = (
            self.agent.joints_controller.art_view.get_joint_velocities().cpu().numpy()
        )  # in degrees / second
        # joint_accelerations
        joint_efforts = (
            self.agent.joints_controller.art_view.get_measured_joint_efforts()
            .cpu()
            .numpy()
        )  # in Nm
        animation_joint_data = self.animation_engine.get_current_clip_datas_ordered(
            phase_signal, joints_order
        )
        animation_joint_angles = animation_joint_data[:, :, 7] / self.reward_space.high
        # animation joint velocities

        from core.utils.math import difference_length_squared_n

        # base position reward
        # base orientation reward
        base_linear_velocity_xy_reward = np.exp(
            difference_length_squared_n(
                base_linear_velocity_xy, np.zeros_like(base_linear_velocity_xy)
            )
            * -8
        )
        base_linear_velocity_z_reward = np.exp(
            np.square(base_linear_velocity_z - np.zeros_like(base_linear_velocity_z))
            * -8
        )
        base_angular_velocity_xy_reward = (
            np.exp(
                difference_length_squared_n(
                    base_angular_velocity_xy, np.zeros_like(base_angular_velocity_xy)
                )
                * -2
            )
            * 0.5
        )
        base_angular_velocity_z_reward = (
            np.exp(
                np.square(
                    base_angular_velocity_z - np.zeros_like(base_angular_velocity_z)
                )
                * -2
            )
            * 0.5
        )
        joint_positions_reward = (
            -difference_length_squared_n(joint_angles, animation_joint_angles) * 15.0
        )
        # joint velocities reward
        joint_efforts_reward = (
            -difference_length_squared_n(joint_efforts, np.zeros_like(joint_efforts))
            * 0.001
        )
        # joint accelerations reward
        joint_action_rate_reward = (
            -difference_length_squared_n(self.actions_buf, self.last_actions_buf) * 1.5
        )
        # joints action acceleration

        self.rewards_buf = (
            base_linear_velocity_xy_reward
            + base_linear_velocity_z_reward
            + base_angular_velocity_xy_reward
            + base_angular_velocity_z_reward
            + joint_positions_reward
            + joint_efforts_reward
            + joint_action_rate_reward
        )

        heights = positions[:, 2]
        self.rewards_buf = np.where(
            heights <= self.reset_height, -20.0, self.rewards_buf
        )
        # self.rewards_buf = np.where(heights >= 1.0, -2.0, self.rewards_buf)

        # TODO: should this be here?
        self.rewards_buf = np.clip(
            self.rewards_buf,
            self.reward_space.low,
            self.reward_space.high,
        )
