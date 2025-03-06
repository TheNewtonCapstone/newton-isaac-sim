from core.universe import Universe


class BaseObject:
    def __init__(
        self,
        universe: Universe,
    ):
        self._universe: Universe = universe

    def __str__(self):
        return f"{self.__class__.__name__}"
