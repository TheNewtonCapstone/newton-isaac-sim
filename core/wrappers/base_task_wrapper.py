from abc import abstractmethod

from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs

from core.tasks import BaseTask
from core.types import Actions


class BaseTaskWrapper(VecEnvWrapper):
    def __init__(self, env: BaseTask):
        super().__init__(env)

        self.venv: BaseTask

    @abstractmethod
    def step_async(self, actions: Actions) -> None:
        super().step_async(actions.cpu().numpy())

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        return super().step_wait()

    @abstractmethod
    def reset(self) -> VecEnvObs:
        return super().reset()
