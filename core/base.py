from typing import Optional

from core.types import Kwargs
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
        assert not self._is_pre_built, f"{self} is already pre-built"

    def post_build(self, **kwargs) -> None:
        assert not self._is_post_built, f"{self} is already post-built"

    def register_self(
        self,
        pre_kwargs: Optional[Kwargs] = None,
        post_kwargs: Optional[Kwargs] = None,
    ) -> None:
        self._universe.register(
            self,
            pre_kwargs=pre_kwargs,
            post_kwargs=post_kwargs,
        )
