from typing import Optional

import torch
from omni.isaac.core.utils.math import normalized
from torch import nn

from core.actuators import BaseActuator
from core.types import (
    VecJointsPositions,
    VecJointsVelocities,
    VecJointsEfforts,
    JointSaturation,
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
        scaler_params: dict,
    ):
        super().__init__(
            universe=universe,
        )

        self._model_path: str = motor_model_path
        self._scaler_params: dict = scaler_params

        self._model: MLPActuatorModel = MLPActuatorModel(
            hidden_size=32,
            num_layers=3,
        ).to(self._universe.device)

        self._position_errors_t0 = None
        self._position_errors_t1 = None
        self._position_errors_t2 = None
        self._velocity_t0 = None
        self._velocity_t1 = None
        self._velocity_t2 = None

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

        self._is_constructed = True

    def post_construct(self) -> None:
        super().post_construct()

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
        output_target_velocities: Optional[VecJointsVelocities] = None,
    ) -> None:
        if output_target_velocities is None:
            output_target_velocities = torch.zeros_like(output_current_velocities)

        # the given positions & velocities are of the output, not the input, which is why we multiply by the gear ratio
        # todo: format the data formating of the _update_efforts to fit current implementation of lstm
        pos_error = (
            output_current_positions - output_target_positions
        ) * self._vec_gear_ratios
        velocities = output_current_velocities * self._vec_gear_ratios
        self._update_history(pos_error, velocities)

        output_position_errors = output_target_positions - output_current_positions
        norm_errors_t0 = normalized(self._position_errors_t0)
        norm_errors_t1 = normalized(self._position_errors_t1)
        norm_errors_t2 = normalized(self._position_errors_t2)

        norm_velocity_t0 = normalized(self._velocity_t0)
        norm_velocity_t1 = normalized(self._velocity_t1)
        norm_velocity_t2 = normalized(self._velocity_t2)

        input_tensor = torch.cat(
            [
                norm_errors_t0,
                norm_errors_t1,
                norm_errors_t2,
                norm_velocity_t0,
                norm_velocity_t1,
                norm_velocity_t2,
            ],
            dim=-1,
        ).unsqueeze(0)

        # input_tensor = torch.cat(
        #     (
        #         output_position_errors * self._vec_gear_ratios,
        #         output_current_velocities * self._vec_gear_ratios,
        #     ),
        #     dim=-1,
        # ).unsqueeze(
        #     0
        # )  # unsqueeze to add batch dimension

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
        self._position_errors_t2 = self._position_errors_t1
        self._position_errors_t1 = self._position_errors_t0
        self._position_errors_t0 = pos_errors
        self._velocity_t2 = self._velocity_t1
        self._velocity_t1 = self._velocity_t0
        self._velocity_t0 = velocity

    def _normalize(self, data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        norm = (data - mean) / std
        return norm

    def _denormalize(self, data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        denorm = (data * std) + mean
        return denorm
