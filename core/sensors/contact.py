from typing import List, Optional

import torch
from torch import Tensor

from omni.isaac.core.prims import RigidPrimView
from ..base import BaseObject
from ..types import ContactData
from ..universe import Universe


class VecContact(BaseObject):
    def __init__(
        self,
        universe: Universe,
        num_contact_sensors_per_agent: int = 1,
    ):
        super().__init__(universe=universe)

        # We type hint universe again here to avoid circular imports
        self._universe: Universe = universe

        self._path_expr: str = ""
        self._paths: List[List[str]] = []
        self._rigid_prim_view: Optional[RigidPrimView] = None

        self._num_contact_sensors_per_agent: int = num_contact_sensors_per_agent
        self._num_contact_sensors: int = 0

        self._last_update_time: float = 0.0

        self._contacts: Tensor = torch.zeros(
            (self._num_contact_sensors, self._num_contact_sensors_per_agent),
            device=self._universe.device,
            dtype=torch.bool,
        )
        self._forces: Tensor = torch.zeros_like(self._contacts, dtype=torch.float)

    @property
    def num_contact_sensors(self) -> int:
        return self._num_contact_sensors

    @property
    def num_contact_sensors_per_agent(self) -> int:
        return self._num_contact_sensors_per_agent

    @property
    def path_expr(self) -> str:
        return self._path_expr

    @property
    def paths(self) -> List[List[str]]:
        return self._paths

    def construct(
        self,
        path_expr: str,
    ) -> None:
        super().construct()

        self._path_expr = path_expr

        from core.utils.usd import find_matching_prims_count

        num_agents = (
            find_matching_prims_count(self._path_expr)
            // self._num_contact_sensors_per_agent
        )

        self._rigid_prim_view = RigidPrimView(
            prim_paths_expr=self._path_expr,
            name="contact_sensor_view",
            track_contact_forces=True,
            disable_stablization=False,
            reset_xform_properties=False,
        )
        self._universe.add_prim(self._rigid_prim_view)

        self._is_constructed = True

    def post_construct(self) -> None:
        super().post_construct()

        from core.utils.usd import find_matching_prims_count

        num_agents = (
            find_matching_prims_count(self._path_expr)
            // self._num_contact_sensors_per_agent
        )

        assert (
            self._rigid_prim_view.count
            == num_agents * self._num_contact_sensors_per_agent
        ), (
            f"Number of contact sensors ({self._rigid_prim_view.count}) does not match the number of agents "
            f"({num_agents}) * number of contact sensors per agent ({self._num_contact_sensors_per_agent})"
        )

        self._num_contact_sensors = (
            self._rigid_prim_view.count // self._num_contact_sensors_per_agent
        )

        self._is_post_constructed = True

        # required to fill the tensors with the correct number of sensors
        self.reset()

    def reset(self) -> None:
        assert (
            self.is_fully_constructed
        ), "Contact sensor not constructed: tried to reset!"

        self._contacts: Tensor = torch.zeros(
            (self._num_contact_sensors, self._num_contact_sensors_per_agent),
            device=self._universe.device,
            dtype=torch.bool,
        )

        self._forces: Tensor = torch.zeros_like(self._contacts, dtype=torch.float)

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

        physics_dt = self._universe.current_time - self._last_update_time
        self._last_update_time = self._universe.current_time

        if physics_dt == 0.0:
            return

        net_forces = self._rigid_prim_view.get_net_contact_forces(dt=physics_dt)
        net_forces = net_forces.view(-1, self._num_contact_sensors_per_agent, 3)

        self._contacts = torch.linalg.norm(net_forces, dim=-1) > 0.0
        self._forces = torch.linalg.norm(net_forces, dim=-1)
