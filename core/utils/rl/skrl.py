from typing import (
    Sequence,
    Union,
    OrderedDict,
    Mapping,
    Any,
    Tuple,
    Optional,
    Type,
)

import torch
from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.trainers.torch import SequentialTrainer
from torch.nn import ReLU

from core.logger import Logger
from core.tasks import BaseTask
from core.types import Config


def create_random_memory(task: BaseTask, memory_size: int) -> Memory:
    from skrl.memories.torch import RandomMemory

    Logger.info(
        f"Creating random memory with task: {task} and memory size: {memory_size}"
    )

    return RandomMemory(
        memory_size=memory_size,
        num_envs=task.num_envs,
        device=task.device,
    )


def create_shared_model(
    task: BaseTask,
    arch: Sequence[int] = (512, 256, 128),
    activation: Type = ReLU,
    is_value_deterministic: bool = True,
) -> Model:
    from skrl.models.torch import GaussianMixin, DeterministicMixin

    POLICY_MIXIN = GaussianMixin
    VALUE_MIXIN = DeterministicMixin if is_value_deterministic else GaussianMixin

    class Shared(POLICY_MIXIN, VALUE_MIXIN, Model):
        def __init__(self):
            Model.__init__(
                self,
                observation_space=task.observation_space,
                action_space=task.action_space,
                device=task.device,
            )
            POLICY_MIXIN.__init__(self, clip_actions=False)
            VALUE_MIXIN.__init__(self, clip_actions=False)

            import torch as th
            from torch import nn

            layers: OrderedDict[str, nn.Module] = OrderedDict[str, nn.Module]()

            # initial layer from the observation space to the first hidden layer
            layers["linear0"] = nn.Linear(task.observation_space.shape[0], arch[0])
            layers["act0"] = activation()

            for i in range(len(arch) - 1):
                layers["linear" + str(i + 1)] = nn.Linear(arch[i], arch[i + 1])
                layers["act" + str(i + 1)] = activation()

            self.net = nn.Sequential(layers)
            self.action_layer = nn.Linear(arch[-1], self.num_actions)
            self.log_std_parameter = nn.Parameter(th.zeros(self.num_actions))
            self.value_layer = nn.Linear(arch[-1], 1)

            self._shared_output: Optional[nn.Sequential] = None

        def act(
            self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "",
        ) -> Tuple[
            torch.Tensor,
            Union[torch.Tensor, None],
            Mapping[str, Union[torch.Tensor, Any]],
        ]:
            if role == "policy":
                return POLICY_MIXIN.act(self, inputs, role)

            if role == "value":
                return VALUE_MIXIN.act(self, inputs, role)

        def compute(
            self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "",
        ) -> Tuple:
            if role == "policy":
                self._shared_output = self.net(inputs["states"])

                return (
                    self.action_layer(self._shared_output),
                    self.log_std_parameter,
                    {},
                )

            if role == "value":
                shared_output = (
                    self.net(inputs["states"])
                    if self._shared_output is None
                    else self._shared_output
                )
                self._shared_output = None

                return self.value_layer(shared_output), {}

    Logger.info(
        f"Creating shared model with task: {task}, architecture: {arch} and activation: {activation}"
    )
    Logger.debug(f" Is value deterministic: {is_value_deterministic}")
    Logger.debug(f" Policy mixin: {POLICY_MIXIN}")
    Logger.debug(f" Value mixin: {VALUE_MIXIN}")

    return Shared()


def create_ppo(
    task: BaseTask,
    ppo_config: Config,
    memory: Memory,
    policy_model: Model,
    value_model: Optional[Model] = None,
    checkpoint_path: Optional[str] = None,
) -> PPO:
    Logger.info(
        f"Creating PPO algorithm with task: {task} and {policy_model}, {value_model} as the policy and value models"
    )
    Logger.debug(f" Memory: {memory}")
    Logger.debug(f" PPO config: {ppo_config}")

    ppo = PPO(
        models={
            "policy": policy_model,
            "value": value_model if value_model is not None else policy_model,
        },
        memory=memory,
        cfg=ppo_config,
        observation_space=task.observation_space,
        action_space=task.action_space,
        device=task.device,
    )

    if checkpoint_path is not None:
        Logger.info(f"Loading PPO checkpoint from {checkpoint_path}")
        ppo.load(checkpoint_path)

    return ppo


def create_sequential_trainer(
    task: BaseTask,
    trainer_config: Config,
    algorithm: Agent,
) -> SequentialTrainer:
    Logger.info(
        f"Creating sequential trainer with task: {task} and algorithm: {algorithm}"
    )
    Logger.debug(f" Trainer config: {trainer_config}")

    return SequentialTrainer(
        agents=[algorithm],
        env=task,  # noqa
        cfg=trainer_config,
    )


def populate_skrl_config(config: Config) -> Config:
    import skrl.resources.preprocessors.torch
    import skrl.resources.schedulers.torch

    Logger.info(f"Populating SKRL config")
    Logger.debug(f" Config: {config}")

    for key, value in config.items():
        if key.endswith("_scheduler"):
            config[key] = getattr(skrl.resources.schedulers.torch, value)
            continue

        if key.endswith("_preprocessor"):
            config[key] = getattr(skrl.resources.preprocessors.torch, value)
            continue

        if key.endswith("_kwargs") and value is None:
            config[key] = {}

    return config
