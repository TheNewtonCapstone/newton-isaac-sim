from typing import Optional

import torch

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


class DCActuator(BaseActuator):
    def __init__(
        self,
        universe: Universe,
        k_p: float,
        k_d: float,
        effort_saturation: JointSaturation,
    ):
        super().__init__(
            universe=universe,
        )

        self._effort_saturation: JointSaturation = effort_saturation

        self._k_p: float = k_p
        self._k_d: float = k_d

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
        input_current_velocities = output_current_velocities * self._vec_gear_ratios

        input_position_errors = (
            output_target_positions - output_current_positions
        ) * self._vec_gear_ratios
        input_velocity_errors = (
            output_target_velocities - output_current_velocities
        ) * self._vec_gear_ratios

        computed_input_efforts = (
            self._k_p * input_position_errors + self._k_d * input_velocity_errors
        )
        # but the computed efforts are from the input's perspective, so we multiply again to get the output's efforts
        self._computed_output_efforts = computed_input_efforts * self._vec_gear_ratios

        applied_input_efforts = self._process_efforts(
            computed_input_efforts,
            input_current_velocities,
        )
        self._applied_output_efforts = applied_input_efforts * self._vec_gear_ratios

    def _process_efforts(
        self,
        input_efforts: VecJointsEfforts,
        input_velocities: VecJointsVelocities,
    ) -> VecJointsEfforts:
        max_effort = self._effort_saturation * (
            1.0 - input_velocities / self._vec_velocity_limits
        )
        max_effort = torch.clamp(
            max_effort,
            min=torch.zeros_like(max_effort),
            max=self._vec_effort_limits,
        )

        min_effort = self._effort_saturation * (
            -1.0 - input_velocities / self._vec_velocity_limits
        )
        min_effort = torch.clamp(
            min_effort,
            min=-self._vec_effort_limits,
            max=torch.zeros_like(min_effort),
        )

        return torch.clamp(input_efforts, min_effort, max_effort)
