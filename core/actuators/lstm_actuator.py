from typing import Optional

import torch
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


class LSTMActuatorModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMActuatorModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Pass through LSTM
        output = self.fc(lstm_out)  # Apply fully connected layer
        return output


class LSTMActuator(BaseActuator):
    def __init__(
        self,
        universe: Universe,
        motor_model_path: str,
    ):
        super().__init__(
            universe=universe,
        )

        self._model_path: str = motor_model_path
        self._model: LSTMActuatorModel = LSTMActuatorModel().to(self._universe.device)

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
        """Computes the efforts to apply to the output joints with a simple PD controller.

        Args:
            output_current_positions: Positions of the joints at the output.
            output_target_positions: Target positions of the joints at the output.
            output_current_velocities: Velocities of the joints at the output.

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
        output_position_errors = output_target_positions - output_current_positions
        input_tensor = torch.cat(
            (
                output_position_errors * self._vec_gear_ratios,
                output_current_velocities * self._vec_gear_ratios,
            ),
            dim=-1,
        ).unsqueeze(
            0
        )  # unsqueeze to add batch dimension

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
