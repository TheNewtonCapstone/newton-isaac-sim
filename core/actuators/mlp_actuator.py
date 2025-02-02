import json
from typing import Optional

import torch
from torch import nn

from core.actuators import BaseActuator
from core.logger import Logger
from core.types import (
    VecJointsPositions,
    VecJointsVelocities,
    VecJointsEfforts,
    VecJointVelocityLimits,
    VecJointEffortLimits,
    VecJointGearRatios,
)
from core.universe import Universe


class MLPActuatorModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=3, output_size=1):
        super(MLPActuatorModel, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.Softsign()]

        # hidden layers
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Softsign()])
            layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize weights"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)


class MLPActuator(BaseActuator):
    def __init__(
        self,
        universe: Universe,
        motor_model_path: str,
        scaler_params_path: str,
    ):
        super().__init__(
            universe=universe,
        )

        self._model_path: str = motor_model_path
        self._scaler_params: dict = json.load(open(scaler_params_path))

        self._model: MLPActuatorModel = MLPActuatorModel(
            hidden_size=32,
            num_layers=3,
        ).to(self._universe.device)

        self._position_errors = torch.zeros(
            (3, self._universe.num_envs),
            device=self._universe.device,
        )
        self._velocities = torch.zeros_like(self._position_errors)

    def construct(
        self,
        input_vec_velocity_limits: VecJointVelocityLimits,
        input_vec_effort_limits: VecJointEffortLimits,
        vec_gear_ratios: VecJointGearRatios,
    ) -> None:
        super().construct(
            input_vec_velocity_limits,
            input_vec_effort_limits,
            vec_gear_ratios,
        )

        self._model.load_state_dict(
            torch.load(self._model_path, map_location=self._universe.device)
        )
        self._model.eval()

        Logger.info(
            f"MLPActuator constructed with model from {self._model_path}, running on {self._universe.device}."
        )

        self._is_constructed = True

    def post_construct(self) -> None:
        super().post_construct()

        Logger.info("MLPActuator post-constructed.")

        self._is_post_constructed = True

    def step(
        self,
        output_current_positions: VecJointsPositions,
        output_target_positions: VecJointsPositions,
        output_current_velocities: VecJointsVelocities,
    ) -> VecJointsEfforts:
        """Compute the effort(torque) to apply to the output joints with an nn
        Args:
            output_target_positions:
            output_target_positions:
            output_current_velocities:
        Returns:

        """
        self._update_efforts(
            output_current_positions,
            output_target_positions,
            output_current_velocities,
        )

        return self._applied_output_efforts

    def reset(self) -> None:
        super().reset()

    def _update_efforts(
        self,
        output_current_positions: VecJointsPositions,
        output_target_positions: VecJointsPositions,
        output_current_velocities: VecJointsVelocities,
    ) -> None:
        output_current_positions = output_current_positions / (2 * torch.pi)
        output_target_positions = output_target_positions / (2 * torch.pi)
        output_current_velocities = output_current_velocities / (2 * torch.pi)

        # the given positions & velocities are of the output, not the input, which is why we multiply by the gear ratio
        input_pos_error = (
            output_target_positions - output_current_positions
        ) * self._vec_gear_ratios
        input_velocities = output_current_velocities * self._vec_gear_ratios

        self._update_history(input_pos_error, input_velocities)

        pos_error_mean = self._scaler_params["pos_error_mean"]
        pos_error_std = self._scaler_params["pos_error_std"]

        vel_mean = self._scaler_params["vel_mean"]
        vel_std = self._scaler_params["vel_std"]

        norm_errors = normalize(self._position_errors, pos_error_mean, pos_error_std)
        norm_velocities = normalize(self._velocities, vel_mean, vel_std)

        input_tensor = torch.cat(
            [
                norm_errors[0, :],
                norm_errors[1, :],
                norm_errors[2, :],
                norm_velocities[0, :],
                norm_velocities[1, :],
                norm_velocities[2, :],
            ],
            dim=-1,
        ).unsqueeze(0)

        with torch.no_grad():
            computed_input_efforts = self._model(input_tensor).squeeze(0)

        # but the computed efforts are from the input's perspective, so we multiply again to get the output's efforts
        self._computed_output_efforts = computed_input_efforts * self._vec_gear_ratios

        applied_input_efforts = self._process_efforts(self._computed_output_efforts)
        self._applied_output_efforts = applied_input_efforts * self._vec_gear_ratios

    def _process_efforts(self, input_efforts: VecJointsEfforts):
        return torch.clamp(
            input_efforts,
            min=-self._vec_effort_limits,
            max=self._vec_effort_limits,
        )

    def _update_history(
        self, pos_errors: VecJointsPositions, velocity: VecJointsVelocities
    ):
        self._position_errors[2, :] = self._position_errors[1, :].clone()
        self._position_errors[1, :] = self._position_errors[0, :].clone()
        self._position_errors[0, :] = pos_errors

        self._velocities[2, :] = self._velocities[1, :].clone()
        self._velocities[1, :] = self._velocities[0, :].clone()
        self._velocities[0, :] = velocity


@torch.jit.script
def normalize(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    norm = (data - mean) / std
    return norm


@torch.jit.script
def denormalize(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    denorm = (data * std) + mean
    return denorm
