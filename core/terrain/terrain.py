# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from typing import List, Optional

import numpy as np

from .generators import (
    pyramid_sloped_terrain,
    pyramid_stairs_terrain,
    random_uniform_terrain,
    discrete_obstacles_terrain,
    stepping_stones_terrain,
    gap_terrain,
    pit_terrain,
)
from .sub_terrain import SubTerrain
from .types import SubTerrainType, TerrainType
from ..base import BaseObject
from ..logger import Logger
from ..types import Config
from ..universe import Universe
from ..utils.terrain import add_heightmap_to_world


class Terrain(BaseObject):
    def __init__(
        self,
        universe: Universe,
        terrain_config: Config,
        num_robots: int,
        root_path: str = "/Terrains",
    ):
        super().__init__(universe=universe)

        self.terrain_config: Config = terrain_config
        self.num_robots: int = num_robots
        self.root_path: str = root_path

        self.mesh_type: str = self.terrain_config["mesh_type"]
        if self.mesh_type in ["none", "plane"]:
            Logger.warning("No terrain mesh will be generated.")
            return

        self.sub_terrain_length: float = self.terrain_config["dimensions"][
            "terrain_length"
        ]
        self.sub_terrain_width: float = self.terrain_config["dimensions"][
            "terrain_width"
        ]

        self.sub_terrain_type_proportion: List[float] = self.terrain_config[
            "generation"
        ]["terrain_proportion"]
        self.sub_terrain_type_proportion = [
            np.sum(self.sub_terrain_type_proportion[: i + 1])
            for i in range(len(self.sub_terrain_type_proportion))
        ]

        self.num_rows: int = self.terrain_config["generation"]["default_num_rows"]
        self.num_cols: int = self.terrain_config["generation"]["default_num_cols"]

        self.vertical_resolution: float = self.terrain_config["generation"][
            "vertical_resolution"
        ]
        self.horizontal_resolution: float = self.terrain_config["generation"][
            "horizontal_resolution"
        ]
        self.slope_threshold: float = self.terrain_config["generation"][
            "slope_threshold"
        ]
        self.sub_terrain_border_size: float = self.terrain_config["dimensions"][
            "border_size"
        ]

        self.sub_terrain_num_width_vertex: int = int(
            self.sub_terrain_width / self.horizontal_resolution
        )
        self.sub_terrain_num_length_vertex: int = int(
            self.sub_terrain_length / self.horizontal_resolution
        )
        self.sub_terrain_num_border_vertex: int = int(
            self.sub_terrain_border_size / self.horizontal_resolution
        )

        self._update_rows_cols_dependents()

    def construct(
        self,
        terrain_type: TerrainType = TerrainType.Curriculum,
        num_rows: Optional[int] = None,
        num_cols: Optional[int] = None,
        sub_terrain_type: Optional[SubTerrainType] = None,
    ) -> None:
        super().construct()

        if num_rows is not None:
            self.num_rows = num_rows
        if num_cols is not None:
            self.num_cols = num_cols

        if num_cols or num_rows:
            self._update_rows_cols_dependents()

        if terrain_type == TerrainType.Random:
            self._construct_randomized()
        elif terrain_type == TerrainType.Specific:
            assert sub_terrain_type is not None, "Sub terrain type must be specified."

            self._construct_selected(terrain_type=sub_terrain_type)
        elif terrain_type == TerrainType.Curriculum:
            self._construct_curriculum()

        if self.mesh_type == "trimesh":
            add_heightmap_to_world(
                self.height_field,
                self.horizontal_resolution,
                self.vertical_resolution,
                self.root_path,
                self.slope_threshold,
                [0, 0, 0],
            )

        self._is_constructed = True

    def post_construct(self) -> None:
        super().post_construct()

        self._is_post_constructed = True

    def _update_rows_cols_dependents(self) -> None:
        self.num_sub_terrains = self.num_rows * self.num_cols
        self.sub_terrain_origins = np.zeros((self.num_rows, self.num_cols, 3))

        self.total_num_rows = (
            int(self.num_rows * self.sub_terrain_num_length_vertex)
            + 2 * self.sub_terrain_num_border_vertex
        )
        self.total_num_cols = (
            int(self.num_cols * self.sub_terrain_num_width_vertex)
            + 2 * self.sub_terrain_num_border_vertex
        )

        self.height_field: np.ndarray = np.zeros(
            (self.total_num_rows, self.total_num_cols),
            dtype=np.int16,
        )

    def _construct_randomized(self):
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.num_rows, self.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self._generate_curriculum_terrains(choice, difficulty)

            self._add_sub_terrain(terrain, i, j)

    def _construct_curriculum(self):
        for j in range(self.num_cols):
            for i in range(self.num_rows):
                difficulty = i / self.num_rows
                choice = j / self.num_cols + 0.001

                terrain = self._generate_curriculum_terrains(choice, difficulty)

                self._add_sub_terrain(terrain, i, j)

    def _construct_selected(self, terrain_type: SubTerrainType, **kwargs):
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.num_rows, self.num_cols))

            terrain = SubTerrain(
                width=self.sub_terrain_num_width_vertex,
                length=self.sub_terrain_num_width_vertex,
                vertical_resolution=self.vertical_resolution,
                horizontal_resolution=self.horizontal_resolution,
            )

            # luckily, the terrain_type enums' names correspond to the function names
            eval(terrain_type.value)(terrain, **kwargs)
            self._add_sub_terrain(terrain, i, j)

    def _generate_curriculum_terrains(self, choice: float, difficulty: float):
        terrain = SubTerrain(
            width=self.sub_terrain_num_width_vertex,
            length=self.sub_terrain_num_width_vertex,
            vertical_resolution=self.vertical_resolution,
            horizontal_resolution=self.horizontal_resolution,
        )

        slope = int(difficulty * 0.4)
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty

        if choice < self.sub_terrain_type_proportion[0]:
            slope *= -1 if choice < self.sub_terrain_type_proportion[0] / 2 else 1

            pyramid_sloped_terrain(
                terrain,
                slope=slope,
                platform_size=3.0,
            )
        elif choice < self.sub_terrain_type_proportion[1]:
            pyramid_sloped_terrain(
                terrain,
                slope=slope,
                platform_size=3.0,
            )
            random_uniform_terrain(
                terrain,
                min_height=-0.05,
                max_height=0.05,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < self.sub_terrain_type_proportion[3]:
            step_height *= -1 if choice < self.sub_terrain_type_proportion[2] else 1

            pyramid_stairs_terrain(
                terrain,
                step_width=0.31,
                step_height=step_height,
                platform_size=3.0,
            )
        elif choice < self.sub_terrain_type_proportion[4]:
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0

            discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
        elif choice < self.sub_terrain_type_proportion[5]:
            stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=4.0,
            )
        elif choice < self.sub_terrain_type_proportion[6]:
            gap_terrain(
                terrain,
                gap_size=gap_size,
                platform_size=3.0,
            )
        else:
            pit_terrain(
                terrain,
                depth=pit_depth,
                platform_size=3.0,
            )

        return terrain

    def _add_sub_terrain(self, terrain: SubTerrain, row: int, col: int):
        i = row
        j = col

        # map coordinate system
        start_x = (
            self.sub_terrain_num_border_vertex + i * self.sub_terrain_num_length_vertex
        )
        end_x = (
            self.sub_terrain_num_border_vertex
            + (i + 1) * self.sub_terrain_num_length_vertex
        )
        start_y = (
            self.sub_terrain_num_border_vertex + j * self.sub_terrain_num_width_vertex
        )
        end_y = (
            self.sub_terrain_num_border_vertex
            + (j + 1) * self.sub_terrain_num_width_vertex
        )

        self.height_field[start_x:end_x, start_y:end_y] = terrain.height_field

        env_origin_x = (i + 0.5) * self.sub_terrain_length
        env_origin_y = (j + 0.5) * self.sub_terrain_width

        x1 = int((self.sub_terrain_length / 2.0 - 1) / terrain.horizontal_resolution)
        x2 = int((self.sub_terrain_length / 2.0 + 1) / terrain.horizontal_resolution)
        y1 = int((self.sub_terrain_width / 2.0 - 1) / terrain.horizontal_resolution)
        y2 = int((self.sub_terrain_width / 2.0 + 1) / terrain.horizontal_resolution)

        env_origin_z = (
            np.max(terrain.height_field[x1:x2, y1:y2]) * terrain.vertical_resolution
        )

        self.sub_terrain_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
