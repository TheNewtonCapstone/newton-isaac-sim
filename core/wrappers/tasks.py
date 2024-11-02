from collections import deque
from random import sample

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs


# inspired from: https://github.com/rmst/rlrd/tree/05a5329066dcabfb7278ec8745890bc2e7edce15
class RandomDelayVecWrapper(VecEnvWrapper):
    """Wrapper for any environment modelling random observation and action delays

    Note that you can access most recent action known to be applied with past_actions[action_delay + observation_delay]
    """

    def __init__(
        self,
        env: VecEnv,
        obs_delay_range=range(0, 8),
        act_delay_range=range(0, 2),
        instant_rewards: bool = True,
    ):
        super().__init__(
            env, observation_space=env.observation_space, action_space=env.action_space
        )

        self.act_delay_range = act_delay_range
        self.obs_delay_range = obs_delay_range
        self.instant_rewards = instant_rewards

        self.past_actions = deque(
            maxlen=obs_delay_range.stop + act_delay_range.stop - 1
        )
        self.past_observations = deque(maxlen=obs_delay_range.stop)
        self.arrival_times_actions = deque(maxlen=act_delay_range.stop)
        self.arrival_times_observations = deque(maxlen=obs_delay_range.stop)

        self.t = 0
        self.current_action = None

    def reset(self) -> VecEnvObs:
        num_wrapped_envs = self.venv.num_envs
        first_observation = self.venv.reset()

        # fill up buffers
        self.t = -(self.obs_delay_range.stop + self.act_delay_range.stop)
        while self.t < 0:
            self.send_action(
                np.zeros(
                    (num_wrapped_envs, *self.venv.action_space.shape),
                    dtype=np.float32,
                )
            )
            self.send_observation(
                (
                    first_observation,
                    np.zeros((num_wrapped_envs,), dtype=np.float32),
                    np.zeros((num_wrapped_envs,), dtype=np.bool_),
                    [{}] * num_wrapped_envs,
                    0,
                )
            )
            self.t += 1

        assert self.t == 0
        received_observation, *_ = self.receive_observation()
        return received_observation

    def step_async(self, action: np.ndarray) -> None:
        # at the brain
        self.send_action(action)

    def step_wait(self) -> VecEnvStepReturn:
        num_wrapped_envs = self.venv.num_envs

        # at the remote actor, for all ifs
        if self.t < self.act_delay_range.stop:
            # do nothing until the brain's first actions arrive at the remote actor
            self.receive_action()
            aux = (
                np.zeros((num_wrapped_envs,), dtype=np.float32),
                np.zeros((num_wrapped_envs,), dtype=np.bool_),
                [{}] * num_wrapped_envs,
            )
        else:
            action_delay = self.receive_action()

            self.venv.step_async(
                self.current_action
            )  # we make sure the wrapped venv knows the action
            m, *aux = self.venv.step_wait()  # we get the results from the wrapped venv

            self.send_observation((m, *aux, action_delay))

        # at the brain again
        m, *delayed_aux = self.receive_observation()
        aux = aux if self.instant_rewards else delayed_aux
        self.t += 1
        return m, *aux

    def send_action(self, action):
        # at the brain
        (delay,) = sample(self.act_delay_range, 1)
        self.arrival_times_actions.appendleft(self.t + delay)
        self.past_actions.appendleft(action)

    def receive_action(self):
        action_delay = next(
            i for i, t in enumerate(self.arrival_times_actions) if t <= self.t
        )
        self.current_action = self.past_actions[action_delay]
        return action_delay

    def send_observation(self, obs):
        # at the remote actor
        (delay,) = sample(self.obs_delay_range, 1)
        self.arrival_times_observations.appendleft(self.t + delay)
        self.past_observations.appendleft(obs)

    def receive_observation(self):
        # at the brain
        observation_delay = next(
            i for i, t in enumerate(self.arrival_times_observations) if t <= self.t
        )
        m, r, d, info, action_delay = self.past_observations[observation_delay]
        return (
            m,
            r,
            d,
            info,
        )
