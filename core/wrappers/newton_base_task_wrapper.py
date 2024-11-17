from abc import abstractmethod

from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from core.tasks import NewtonBaseTask
from core.types import Actions
from core.wrappers import BaseTaskWrapper


class NewtonBaseTaskWrapper(BaseTaskWrapper):
    def __init__(self, env: NewtonBaseTask):
        super().__init__(env)

        self.venv: NewtonBaseTask

    @abstractmethod
    def step_async(self, actions: Actions) -> None:
        return super().step_async(actions)

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        return super().step_wait()

    @abstractmethod
    def reset(self) -> VecEnvObs:
        return super().reset()
