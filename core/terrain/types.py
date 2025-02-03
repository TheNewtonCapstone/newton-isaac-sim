from enum import IntEnum, Enum


class SubTerrainType(Enum):
    PyramidSloped = "pyramid_sloped_terrain"
    PyramidStairs = "pyramid_stairs_terrain"
    RandomUniform = "random_uniform_terrain"
    DiscreteObstacles = "discrete_obstacles_terrain"
    SteppingStones = "stepping_stones_terrain"
    GapTerrain = "gap_terrain"
    PitTerrain = "pit_terrain"


class TerrainType(IntEnum):
    Random = 0
    Specific = 1
    Curriculum = 2
