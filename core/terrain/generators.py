import numpy as np

from .sub_terrain import SubTerrain
from scipy import interpolate


def random_uniform_terrain(
    terrain: SubTerrain,
    min_height: float = 0.0,
    max_height: float = 0.05,
    step: float = 0.01,
    downsampled_scale: float = None,
) -> SubTerrain:
    """
    Generates a uniform noise terrain.

    Parameters
        terrain: Terrain to modify.
        min_height: The minimum height of the terrain (in meters).
        max_height: The maximum height of the terrain (in meters).
        step: Minimum height change between two points (in meters).
        downsampled_scale: Distance between two randomly sampled points (must be larger or equal to terrain.horizontal_scale).

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_resolution

    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_resolution)
    max_height = int(max_height / terrain.vertical_resolution)
    step = int(step / terrain.vertical_resolution)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(
        heights_range,
        (
            int(terrain.width * terrain.horizontal_resolution / downsampled_scale),
            int(terrain.length * terrain.horizontal_resolution / downsampled_scale),
        ),
    )

    x = np.linspace(
        0,
        terrain.width * terrain.horizontal_resolution,
        height_field_downsampled.shape[0],
    )
    y = np.linspace(
        0,
        terrain.length * terrain.horizontal_resolution,
        height_field_downsampled.shape[1],
    )

    f = interpolate.RectBivariateSpline(y, x, height_field_downsampled)

    x_upsampled = np.linspace(
        0, terrain.width * terrain.horizontal_resolution, terrain.width
    )
    y_upsampled = np.linspace(
        0, terrain.length * terrain.horizontal_resolution, terrain.length
    )
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain.height_field += z_upsampled.astype(np.int16)

    return terrain


def sloped_terrain(terrain: SubTerrain, slope: int = 1) -> SubTerrain:
    """
    Generates a sloped terrain.

    Parameters:
        terrain: Terrain to modify.
        slope: Dictates the slope of the terrain (positive or negative).
    Returns:
        terrain: Sloped terrain.
    """

    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.width, 1)

    max_height = int(
        slope
        * (terrain.horizontal_resolution / terrain.vertical_resolution)
        * terrain.width
    )

    terrain.height_field[:, np.arange(terrain.length)] += (
        max_height * xx / terrain.width
    ).astype(terrain.height_field.dtype)

    return terrain


def pyramid_sloped_terrain(
    terrain: SubTerrain,
    slope: int = 1,
    platform_size: float = 1.0,
) -> SubTerrain:
    """
    Generates a sloped terrain.

    Parameters:
        terrain: Terrain to modify.
        slope: Dictates the slope of the terrain (positive or negative).
        platform_size: size of the flat platform at the center of the terrain (in meters)
    Returns:
        terrain: Pyramid sloped terrain.
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)

    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)

    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y

    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)

    max_height = int(
        slope
        * (terrain.horizontal_resolution / terrain.vertical_resolution)
        * (terrain.width / 2)
    )

    terrain.height_field += (max_height * xx * yy).astype(terrain.height_field.dtype)

    platform_size = int(platform_size / terrain.horizontal_resolution / 2)
    x1 = terrain.width // 2 - platform_size
    x2 = terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    y2 = terrain.length // 2 + platform_size

    min_h = min(terrain.height_field[x1, y1], 0)
    max_h = max(terrain.height_field[x1, y1], 0)
    terrain.height_field = np.clip(terrain.height_field, min_h, max_h)

    return terrain


def discrete_obstacles_terrain(
    terrain: SubTerrain,
    max_height: float = 1.0,
    min_size: float = 0.05,
    max_size: float = 0.2,
    num_rects: int = 5,
    platform_size=1.0,
) -> SubTerrain:
    """
    Generates a terrain with rectangular obstacles.

    Parameters:
        terrain: Terrain to modify.
        max_height: Maximum height of the obstacles (range=[-max, -max/2, max/2, max]; in meters).
        min_size: Minimum size of a rectangle obstacle (in meters).
        max_size: Maximum size of a rectangle obstacle (in meters).
        num_rects: Number of randomly generated obstacles.
        platform_size: Size of the flat platform at the center of the terrain (in meters).
    Returns:
        terrain: Terrain with rectangular obstacles.
    """

    # switch parameters to discrete units
    max_height = int(max_height / terrain.vertical_resolution)
    min_size = int(min_size / terrain.horizontal_resolution)
    max_size = int(max_size / terrain.horizontal_resolution)
    platform_size = int(platform_size / terrain.horizontal_resolution)

    (i, j) = terrain.height_field.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i - width, 4))
        start_j = np.random.choice(range(0, j - length, 4))

        terrain.height_field[start_i : start_i + width, start_j : start_j + length] = (
            np.random.choice(height_range)
        )

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2

    terrain.height_field[x1:x2, y1:y2] = 0

    return terrain


def wave_terrain(
    terrain: SubTerrain,
    num_waves: int = 1,
    amplitude: float = 1.0,
) -> SubTerrain:
    """
    Generates a wavy terrain.

    Parameters:
        terrain: Terrain to modify.
        num_waves: Number of sine waves across the terrain length.
        amplitude: Amplitude of the sine waves.
    Returns:
        terrain: Wavy terrain.
    """
    amplitude = int(0.5 * amplitude / terrain.vertical_resolution)

    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width, 1)
        yy = yy.reshape(1, terrain.length)

        terrain.height_field += (
            amplitude * np.cos(yy / div) + amplitude * np.sin(xx / div)
        ).astype(terrain.height_field.dtype)

    return terrain


def stairs_terrain(
    terrain: SubTerrain,
    step_width: float = 0.2,
    step_height: float = 0.1,
) -> SubTerrain:
    """
    Generates stairs.

    Parameters:
        terrain: Terrain to modify.
        step_width: The width of the step (in meters).
        step_height: The height of the step (in meters).
    Returns:
        terrain: Stairs terrain.
    """

    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_resolution)
    step_height = int(step_height / terrain.vertical_resolution)

    num_steps = terrain.width // step_width
    height = step_height

    for i in range(num_steps):
        terrain.height_field[i * step_width : (i + 1) * step_width, :] += height
        height += step_height

    return terrain


def pyramid_stairs_terrain(
    terrain: SubTerrain,
    step_width: float = 0.2,
    step_height: float = 0.1,
    platform_size: float = 1.0,
) -> SubTerrain:
    """
    Generates stairs with a pyramid shape.

    Parameters:
        terrain: Terrain to modify.
        step_width: The width of the step (in meters).
        step_height: The step_height (in meters).
        platform_size: Size of the flat platform at the center of the terrain (in meters).
    Returns:
        terrain: Pyramid stairs terrain.
    """

    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_resolution)
    step_height = int(step_height / terrain.vertical_resolution)
    platform_size = int(platform_size / terrain.horizontal_resolution)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length

    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field[start_x:stop_x, start_y:stop_y] = height

    return terrain


def stepping_stones_terrain(
    terrain: SubTerrain,
    stone_size: float = 0.2,
    stone_distance: float = 0.2,
    max_height: float = 0.5,
    platform_size: float = 1.0,
    depth: float = -10.0,
) -> SubTerrain:
    """
    Generates a stepping stones terrain.

    Parameters:
        terrain: Terrain to modify.
        stone_size: Horizontal size of the stepping stones (in meters).
        stone_distance: Distance between stones (i.e size of the holes; in meters).
        max_height: Maximum height of the stones (positive and negative; in meters).
        platform_size: Size of the flat platform at the center of the terrain (in meters).
        depth: Depth of the holes (default=-10.0; in meters).
    Returns:
        terrain (SubTerrain): update terrain
    """

    # switch parameters to discrete units
    stone_size = max(int(stone_size / terrain.horizontal_resolution), 1)
    stone_distance = int(stone_distance / terrain.horizontal_resolution)
    max_height = int(max_height / terrain.vertical_resolution)
    platform_size = int(platform_size / terrain.horizontal_resolution)
    height_range = np.arange(-max_height - 1, max_height, step=1)

    start_x = 0
    start_y = 0

    terrain.height_field[:, :] = int(depth / terrain.vertical_resolution)

    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)

            # fill first hole
            stop_x = max(0, start_x - stone_distance)
            terrain.height_field[0:stop_x, start_y:stop_y] = np.random.choice(
                height_range
            )
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field[start_x:stop_x, start_y:stop_y] = np.random.choice(
                    height_range
                )

                start_x += stone_size + stone_distance

            start_y += stone_size + stone_distance
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)

            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field[start_x:stop_x, 0:stop_y] = np.random.choice(
                height_range
            )

            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field[start_x:stop_x, start_y:stop_y] = np.random.choice(
                    height_range
                )

                start_y += stone_size + stone_distance

            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2

    terrain.height_field[x1:x2, y1:y2] = 0

    return terrain


def gap_terrain(
    terrain: SubTerrain,
    gap_size: float = 0.25,
    platform_size: float = 1.0,
) -> SubTerrain:
    """
    Generates a terrain with a gap around the center.

    Args:
        terrain: Terrain to modify.
        gap_size: Size of the gap (in meters).
        platform_size: Size of the flat platform at the center of the terrain (in meters).

    Returns:
        terrain: Terrain with a gap around the center.
    """
    gap_size = int(gap_size / terrain.horizontal_resolution)
    platform_size = int(platform_size / terrain.horizontal_resolution)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0

    return terrain


def pit_terrain(
    terrain: SubTerrain,
    depth: float = 1.0,
    platform_size: float = 1.0,
) -> SubTerrain:
    """
    Generates a pit in the center of the terrain.

    Args:
        terrain: Terrain to modify.
        depth: Depth of the pit (in meters).
        platform_size: Size of the flat platform at the center of the terrain (in meters).

    Returns:
        terrain: Terrain with a pit in the center.
    """

    depth = int(depth / terrain.vertical_resolution)
    platform_size = int(platform_size / terrain.horizontal_resolution / 2)

    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size

    terrain.height_field[x1:x2, y1:y2] = -depth

    return terrain
