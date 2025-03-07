from core.universe import Universe


class BaseObject:
    def __init__(
        self,
        universe: Universe,
    ):
        self._universe: Universe = universe

        self._is_pre_built: bool = False
        self._is_post_built: bool = False

    def __str__(self):
        return f"{self.__class__.__name__}"

    @property
    def is_pre_built(self) -> bool:
        return self._is_pre_built

    @property
    def is_post_built(self) -> bool:
        return self._is_post_built

    @property
    def is_built(self) -> bool:
        return self._is_pre_built and self._is_post_built

    def pre_build(self, **kwargs) -> None:
        pass

    def post_build(self, **kwargs) -> None:
        pass

    def register_self(self, pre_kwargs: Dict[str, Any] = None, Dict[str, Any]) -> None:
        pass
