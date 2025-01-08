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
        assert (
            self.is_fully_constructed
        ), "Actuator not fully constructed: tried to access computed efforts!"

        return self._computed_output_efforts

    @property
    def applied_output_efforts(self) -> VecJointsEfforts:
        """Returns applied efforts, after any clamping, saturation, etc. and, most importantly, after the gear ratio."""
        assert (
            self.is_fully_constructed
        ), "Actuator not fully constructed: tried to access applied efforts!"

        return self._applied_output_efforts

    @abstractmethod
    def construct(
        self,
        input_vec_velocity_limits: VecJointVelocityLimits,
        input_vec_effort_limits: VecJointEffortLimits,
        vec_gear_ratios: VecJointVelocityLimits,
    ) -> None:
        super().construct()

        # limits are given of the input
        self._vec_velocity_limits = input_vec_velocity_limits
        self._vec_effort_limits = input_vec_effort_limits

        self._vec_gear_ratios = vec_gear_ratios

    @abstractmethod
    def post_construct(self) -> None:
        super().post_construct()

    @abstractmethod
    def step(
        self,
        output_current_positions: VecJointsPositions,
        output_target_positions: VecJointsPositions,
        output_current_velocities: VecJointsVelocities,
    ) -> VecJointsEfforts:
        assert (
            self.is_fully_constructed
        ), "Actuator not fully constructed: tried to step!"

        return self._applied_output_efforts

    @abstractmethod
    def reset(self) -> None:
        assert (
            self.is_fully_constructed
        ), "Actuator not fully constructed: tried to reset!"

        self._computed_output_efforts = torch.zeros_like(self._computed_output_efforts)
        self._applied_output_efforts = torch.zeros_like(self._applied_output_efforts)
