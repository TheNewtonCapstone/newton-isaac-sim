from abc import abstractmethod

import torch

from ..base import BaseObject
from ..types import (
    VecJointsEfforts,
    VecJointsPositions,
    VecJointsVelocities,
    VecJointEffortLimits,
    VecJointVelocityLimits,
    VecJointGearRatios,
)
from ..universe import Universe


class BaseActuator(BaseObject):
    def __init__(
        self,
        universe: Universe,
    ):
        super().__init__(universe=universe)

        num_envs: int = self._universe.num_envs

        self._vec_velocity_limits: VecJointVelocityLimits = torch.zeros(
            (num_envs,),
            device=self._universe.device,
        )
        self._vec_effort_limits: VecJointEffortLimits = torch.zeros_like(
            self._vec_velocity_limits,
        )

        self._vec_gear_ratios: VecJointGearRatios = torch.zeros_like(
            self._vec_velocity_limits,
        )

        self._target_positions: VecJointsPositions = torch.zeros_like(
            self._vec_velocity_limits,
        )
        self._computed_output_efforts: VecJointsEfforts = torch.zeros_like(
            self._vec_velocity_limits,
        )
        self._applied_output_efforts: VecJointsEfforts = torch.zeros_like(
            self._vec_velocity_limits,
        )

    @property
    def computed_output_efforts(self) -> VecJointsEfforts:
        """Returns computed efforts, after any clamping, saturation, etc. and, most importantly, after the gear ratio."""
        return self._computed_output_efforts

    @property
    def applied_output_efforts(self) -> VecJointsEfforts:
        """Returns applied efforts, after any clamping, saturation, etc. and, most importantly, after the gear ratio."""
        return self._applied_output_efforts

    @abstractmethod
    def build(
        self,
        output_vec_velocity_limits: VecJointVelocityLimits,
        output_vec_effort_limits: VecJointEffortLimits,
        vec_gear_ratios: VecJointVelocityLimits,
    ) -> None:
        # limits of the input, in rad/s and Nm
        self._vec_velocity_limits = output_vec_velocity_limits * vec_gear_ratios
        self._vec_effort_limits = output_vec_effort_limits / vec_gear_ratios

        self._vec_gear_ratios = vec_gear_ratios

    @abstractmethod
    def step(
        self,
        output_current_positions: VecJointsPositions,
        output_target_positions: VecJointsPositions,
        output_current_velocities: VecJointsVelocities,
    ) -> VecJointsEfforts:
        """Computes the efforts to apply to the output joints with a simple PD controller.

        Args:
            output_current_positions: Positions of the joints at the output (in rad).
            output_target_positions: Target positions of the joints at the output (in rad).
            output_current_velocities: Velocities of the joints at the output (in rad/s).

        Returns:
            VecJointsEfforts: Efforts to apply to the output joints (in Nm).

        """
        return self._applied_output_efforts

    @abstractmethod
    def reset(self) -> None:
        self._computed_output_efforts = torch.zeros_like(self._computed_output_efforts)
        self._applied_output_efforts = torch.zeros_like(self._applied_output_efforts)
