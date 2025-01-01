from abc import abstractmethod

import torch

from core.types import (
    VecJointsEfforts,
    VecJointsPositions,
    VecJointsVelocities,
    VecJointEffortLimits,
    VecJointVelocityLimits,
    VecJointGearRatios,
)
from core.universe import Universe


class BaseActuator:
    def __init__(
        self,
        universe: Universe,
    ):
        """
        Args:
            universe: Unique instance of the Universe class.
            gear_ratio: Gear ratio of the actuator, so if the motor spins `gear_ratio` revolutions, the output spins 1 revolution.
        """
        self._universe: Universe = universe

        self._vec_velocity_limits: VecJointVelocityLimits = torch.zeros(
            (0,),
            device=self._universe.device,
        )
        self._vec_effort_limits: VecJointEffortLimits = torch.zeros_like(
            self._vec_velocity_limits,
        )

        self._vec_gear_ratios: VecJointGearRatios = torch.zeros_like(
            self._vec_velocity_limits,
        )

        self._target_positions: VecJointsPositions = torch.zeros(
            (0,),
            device=self._universe.device,
        )
        self._computed_output_efforts: VecJointsEfforts = torch.zeros_like(
            self._target_positions,
        )
        self._applied_output_efforts: VecJointsEfforts = torch.zeros_like(
            self._target_positions,
        )

        self._is_constructed: bool = False

    @property
    def computed_output_efforts(self) -> VecJointsEfforts:
        """Returns computed efforts, after any clamping, saturation, etc. and, most importantly, after the gear ratio."""
        assert (
            self._is_constructed
        ), "Actuator not constructed: tried to access computed efforts!"

        return self._computed_output_efforts

    @property
    def applied_output_efforts(self) -> VecJointsEfforts:
        """Returns applied efforts, after any clamping, saturation, etc. and, most importantly, after the gear ratio."""
        assert (
            self._is_constructed
        ), "Actuator not constructed: tried to access applied efforts!"

        return self._applied_output_efforts

    @abstractmethod
    def construct(
        self,
        input_vec_velocity_limits: VecJointVelocityLimits,
        input_vec_effort_limits: VecJointEffortLimits,
        vec_gear_ratios: VecJointVelocityLimits,
    ) -> None:
        assert (
            not self._is_constructed
        ), "Actuator already constructed: tried to construct!"

        # limits are given of the input
        self._vec_velocity_limits = input_vec_velocity_limits
        self._vec_effort_limits = input_vec_effort_limits

        self._vec_gear_ratios = vec_gear_ratios

        self._is_constructed = True

    @abstractmethod
    def step(
        self,
        output_current_positions: VecJointsPositions,
        output_target_positions: VecJointsPositions,
        output_current_velocities: VecJointsVelocities,
    ) -> VecJointsEfforts:
        assert self._is_constructed, "Actuator not constructed: tried to step!"

        return self._applied_output_efforts

    @abstractmethod
    def reset(self) -> None:
        assert self._is_constructed, "Actuator not constructed: tried to reset!"

        self._computed_output_efforts = torch.zeros_like(self._computed_output_efforts)
        self._applied_output_efforts = torch.zeros_like(self._applied_output_efforts)
