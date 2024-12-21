from typing import Optional

from core.types import Settings
from isaacsim import SimulationApp

_universe: Optional["Universe"] = None


def big_bang(
    app_settings: Settings,
    world_settings: Settings,
    experience: str = "./apps/omni.isaac.sim.newton.kit",
) -> "Universe":
    """
    Initializes the universe and the Omniverse simulation app. Can be called multiple times.
    Args:
        app_settings: The settings for the simulation app (i.e. rendering).
        world_settings: The settings for the world (i.e. physics).
        experience: The path to the experience to load (Omniverse Kit).

    Returns:
        The Universe instance. If called multiple times, returns the same instance.
    """
    global _universe

    if _universe is not None:
        return _universe

    sim_app: SimulationApp = SimulationApp(app_settings, experience)

    from core.universe import Universe

    _universe = Universe(app_settings["headless"], sim_app, world_settings)
    _universe.reset(soft=False)

    return _universe
