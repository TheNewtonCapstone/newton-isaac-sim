from typing import Dict, Any, Tuple, Callable

import gymnasium
import numpy as np
import torch
from core.base.base_agent import BaseAgent
from core.base.base_env import BaseEnv
from core.base.base_task import BaseTask
from core.utils.math import gaussian_distribute
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs


class BalancingTwipCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=2)

    def _on_step(self) -> bool:
        task: BalancingTwipTask = self.training_env

        self.logger.record("rewards/mean", task.rewards_buf.mean().item())
        self.logger.record("rewards/median", torch.median(task.rewards_buf).item())
        self.logger.record("rewards/sum", task.rewards_buf.sum().item())

        self.logger.record("metrics/roll_mean", task.obs_buf[:, 0].mean().item())
        self.logger.record(
            "metrics/ang_vel_roll_mean", task.obs_buf[:, 1].mean().item()
        )
        self.logger.record("metrics/ang_vel_yaw_mean", task.obs_buf[:, 2].mean().item())

        # combined_actions = torch.sum(torch.abs(task.actions_buf), dim=-1)
        # self.logger.record(
        #    "metrics/action_magnitude_mean", combined_actions.mean().item()
        # )

        # save the model intermittently
        if self.model.num_timesteps % (512 * task.num_envs) == 0:
            self.model.save(f"{self.logger.dir}/model_{self.model.num_timesteps}.zip")

        return True


class BalancingTwipTask(BaseTask):
    def __init__(
        self,
        headless: bool,
        device: str,
        num_envs: int,
        playing: bool,
        max_episode_length: int,
        training_env_factory: Callable[..., BaseEnv],
        playing_env_factory: Callable[..., BaseEnv],
        agent_factory: Callable[..., BaseAgent],
    ):
        observation_space = gymnasium.spaces.Box(
            low=np.array(
                [
                    -np.pi,
                    -np.Inf,
                    -np.Inf,
                    # -1.0,
                    # -1.0,
                ]
            ),
            high=np.array(
                [
                    np.pi,
                    np.Inf,
                    np.Inf,
                    # 1.0,
                    # 1.0,
                ]
            ),
        )

        num_actions = 2
        action_space = gymnasium.spaces.Box(
            low=np.ones(num_actions) * -1.0,
            high=np.ones(num_actions) * 1.0,
        )

        reward_space = gymnasium.spaces.Box(
            low=np.array([-2.0]),
            high=np.array([1.0]),
        )

        super().__init__(
            headless=headless,
            device=device,
            num_envs=num_envs,
            playing=playing,
            max_episode_length=max_episode_length,
            observation_space=observation_space,
            action_space=action_space,
            reward_space=reward_space,
            training_env_factory=training_env_factory,
            playing_env_factory=playing_env_factory,
            agent_factory=agent_factory,
        )

    def construct(self) -> bool:
        self.env = (
            self.playing_env_factory() if self.playing else self.training_env_factory()
        )
        self.agent = self.agent_factory()

        self.env.construct(self.agent)

        return True

    # Gymnasium methods (required from VecEnv)

    def reset(self) -> VecEnvObs:
        # resets the entire task (i.e. beginning of training/playing)
        # returns: the observations

        super().reset()

        self.env.reset()

        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations), dtype=torch.float32
        )

        return self.obs_buf.numpy().copy()

    def step_wait(self) -> VecEnvStepReturn:
        if self.actions_buf is None:
            return (
                self.obs_buf.numpy().copy(),
                self.rewards_buf.numpy().copy(),
                self.dones_buf.numpy().copy(),
                [],
            )

        self.progress_buf += 1

        # Ensure that the actions are within the action space
        self.actions_buf = torch.clamp(
            self.actions_buf,
            torch.from_numpy(self.action_space.low),
            torch.from_numpy(self.action_space.high),
        )

        # shape of twip_imu_obs: (num_envs, 10)
        # (:, 0:3) -> linear acceleration
        # (:, 3:6) -> angular velocity
        # (:, 6:10) -> quaternion (WXYZ)
        twip_imu_obs = self.env.step(
            actions_to_torque(gaussian_distribute(self.actions_buf)),
            render=not self.headless,
        )

        twip_roll = roll_from_quat(twip_imu_obs[:, 6:10])
        twip_roll_velocity = twip_imu_obs[:, 3]
        twip_yaw_velocity = twip_imu_obs[:, 5]

        self.obs_buf[:, 0] = gaussian_distribute(twip_roll)
        self.obs_buf[:, 1] = gaussian_distribute(twip_roll_velocity)
        self.obs_buf[:, 2] = gaussian_distribute(twip_yaw_velocity)
        # self.obs_buf[:, 3] = self.actions_buf[:, 0]
        # self.obs_buf[:, 4] = self.actions_buf[:, 1]

        # the smaller the difference between current orientation and stable orientation, the higher the reward
        self.rewards_buf = compute_rewards_twip(
            twip_roll, twip_roll_velocity, twip_yaw_velocity, self.actions_buf
        )

        # Clip rewards to the specified range
        self.rewards_buf = torch.clamp(
            self.rewards_buf,
            torch.from_numpy(self.reward_space.low),
            torch.from_numpy(self.reward_space.high),
        )

        # process failures (when falling) # TODO: check if this is being used correctly
        self.dones_buf = torch.where(
            torch.abs(twip_roll) > 0.5, True, False
        )  # terminated
        self.dones_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1, True, self.dones_buf
        )  # truncated

        # creates a new tensor with only the indices of the environments that are done
        resets = self.dones_buf.nonzero(as_tuple=False).flatten()
        if len(resets) > 0:
            self.env.reset(resets)

        # clears the last 2 observations & the progress if any twip is reset
        self.obs_buf[resets, :] = 0.0
        self.progress_buf[resets] = 0

        return (
            self.obs_buf.numpy().copy(),
            self.rewards_buf.numpy().copy(),
            self.dones_buf.numpy().copy(),
            [{} for _ in range(self.num_envs)],
        )


@torch.jit.script
def pwm_percent_to_torque(pwm_percent: torch.Tensor) -> torch.Tensor:
    # if any pwm_percent is less than 0.375, return 0
    # otherwise, return the torque
    # this follows empirical data
    return 0.653 + 0.507 * torch.log(pwm_percent)


@torch.jit.script
def actions_to_torque(actions: torch.Tensor) -> torch.Tensor:
    # store the sign for later (for directions)
    actions_sign = torch.sign(actions)

    # map the actions to the PWM range [0.375, 1.0]
    actions = torch.abs(actions)
    actions = actions * 0.625 + 0.375

    # convert the PWM to torque
    actions = pwm_percent_to_torque(actions)

    # apply the sign, for directions
    actions = actions * actions_sign

    return actions


@torch.jit.script
def compute_rewards_twip(
    roll: torch.Tensor,
    ang_vel_x: torch.Tensor,
    ang_vel_z: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    # computes the rewards based on the roll, angular velocity and actions
    # args: roll, angular velocity, actions
    # returns: rewards

    # Compute rate of change of roll
    roll_velocity = ang_vel_x

    # Normalized position (assume max distance is 5 units)
    # norm_position = torch.norm(position, dim=-1) / 5.0

    # Penalties
    roll_penalty = torch.tanh(6 * torch.abs(roll))
    roll_velocity_penalty = torch.tanh(torch.abs(roll_velocity)) * 0.2
    ang_vel_penalty = torch.tanh(2 * torch.abs(ang_vel_z)) * 0.8
    # action_penalty = (
    #    torch.tanh(torch.sum(torch.square(actions_to_torque(actions)), dim=-1)) * 0.2
    # )
    # position_penalty = torch.tanh(norm_position) * 0.2
    # roll_velocity_penalty = torch.tanh(2 * torch.abs(roll_velocity)) * 0.3

    # Compute rewards
    rewards = (
        1.0
        - roll_penalty
        - roll_velocity_penalty
        - ang_vel_penalty
        # - action_penalty
        # - position_penalty
        # - roll_velocity_penalty
    )

    # Time factor to encourage longer balancing (assuming time_step is in seconds)
    # time_factor = torch.tanh(time_step / 10.0)  # Normalize to [0, 1] over 10 seconds
    # rewards *= 1.0 + time_factor

    # Harsh penalty for falling
    rewards = torch.where(torch.abs(roll) > 0.5, -2.0, rewards)

    return rewards


@torch.jit.script
def roll_from_quat(q: torch.Tensor) -> torch.Tensor:
    # extracts from a quaternion, the roll in rads, expects a 2dim tensor
    # args: quaternion
    # returns: euler angles

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    roll = torch.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    # pitch = torch.arcsin(2 * (w * y - z * x))
    # yaw = torch.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return roll
