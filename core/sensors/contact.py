from typing import Dict, List, Optional

import torch
from core.types import ContactData
from core.universe import Universe
from torch import Tensor


class VecContact:
    def __init__(
        self,
        universe: Universe,
        num_contact_sensors_per_agent: int = 1,
    ):
        from omni.isaac.sensor import ContactSensor

        self.universe: Universe = universe
        self.path_expr: str = ""
        self.transform_path_expr: str = ""
        self.paths: List[List[str]] = []
        self.transform_paths: List[List[str]] = []
        self.sensors: List[ContactSensor] = []

        self.num_contact_sensors_per_agent: int = num_contact_sensors_per_agent
        self.num_contact_sensors: int = 0

        self._sensor_interface: Optional = None
        self._last_update_time: float = 0.0

        self._is_constructed: bool = False

        self._contacts: Tensor = torch.zeros(
            (self.num_contact_sensors, self.num_contact_sensors_per_agent),
            device=self.universe.device,
            dtype=torch.bool,
        )
        self._forces: Tensor = torch.zeros_like(self._contacts, dtype=torch.float)

    def construct(
        self,
        sensor_name: str,
        body_path_expr: str,
        relative_transform_path_expr: str,
    ) -> None:
        assert (
            not self._is_constructed
        ), "Contact sensor already constructed: tried to construct!"

        from core.utils.usd import (
            path_expr_to_paths,
            get_parent_expr_path_from_expr_path,
            get_parent_path_from_path,
        )

        self.transform_path_expr = body_path_expr + relative_transform_path_expr
        self.path_expr = (
            get_parent_expr_path_from_expr_path(self.transform_path_expr)
            + f"/{sensor_name}"
        )

        self.transform_paths = path_expr_to_paths(
            body_path_expr,
            relative_transform_path_expr,
        )

        from omni.isaac.sensor import ContactSensor
        from omni.isaac.core.utils.prims import get_prim_attribute_value

        for i, children_transform_paths in enumerate(self.transform_paths):
            self.paths.append([""] * len(children_transform_paths))

            for j, transform_path in enumerate(children_transform_paths):
                sensor_path = (
                    get_parent_path_from_path(transform_path) + f"/{sensor_name}"
                )

                sensor_translation = get_prim_attribute_value(
                    prim_path=transform_path,
                    attribute_name="xformOp:translate",
                )

                sensor = ContactSensor(
                    prim_path=sensor_path,
                    name=f"{sensor_name}_{i}_{j}",
                    translation=sensor_translation,
                    radius=0.03,
                )
                self.universe.add(sensor)

                self.sensors.append(sensor)
                self.paths[i][j] = sensor_path

        assert (
            len(self.paths[0]) == self.num_contact_sensors_per_agent
        ), "Given number of contact sensors per agent does not match the number found!"

        # propagate physics changes
        self.universe.reset()

        self.num_contact_sensors = len(self.paths)

        self._is_constructed = True

        # required to fill the tensors with the correct number of IMUs
        self.reset()

    def reset(self) -> ContactData:
        self._contacts: Tensor = torch.zeros(
            (self.num_contact_sensors, self.num_contact_sensors_per_agent),
            device=self.universe.device,
            dtype=torch.bool,
        )

        self._forces: Tensor = torch.zeros_like(self._contacts, dtype=torch.float)

        return self.get_data()

    def get_data(self) -> ContactData:
        raw_data = self.get_raw_data()

        return raw_data

    def get_raw_data(self) -> ContactData:
        self._update_data()

        return {
            "in_contacts": self._contacts,
            "contact_forces": self._forces,
        }

    def _update_data(self) -> None:
        assert (
            self._is_constructed
        ), "Contact sensor not constructed: tried to update data!"

        for i, sensor in enumerate(self.sensors):
            reading: Dict = sensor.get_current_frame()

            x = i // self.num_contact_sensors_per_agent
            y = i % self.num_contact_sensors_per_agent

            self._contacts[x, y] = reading["in_contact"]
            self._forces[x, y] = reading["force"]
