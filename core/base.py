from abc import abstractmethod

# TODO: Add `BaseSteppable` and `BaseResettable` classes
#   They would inherit from `BaseObject` and have `step` and `reset` methods, respectively. The goal is to standardize
#   the interface as much as possible (for clarity and ease of use).


class BaseObject:
    def __init__(
        self,
        universe: "Universe",
    ):
        self._universe: "Universe" = universe

        self._is_constructed: bool = False
        self._is_post_constructed: bool = False

    def __str__(self):
        return f"{self.__class__.__name__}"

    @property
    def is_constructed(self):
        return self._is_constructed

    @property
    def is_post_constructed(self):
        return self._is_post_constructed

    @property
    def is_fully_constructed(self):
        return self._is_constructed and self._is_post_constructed

    def register_self(self, *args, **kwargs) -> None:
        self._universe.register_object(self, *args, **kwargs)

    @abstractmethod
    def construct(self, *args) -> None:
        assert (
            not self._is_constructed
        ), f"{self} has already been constructed: tried to construct!"

    @abstractmethod
    def post_construct(self, **kwargs) -> None:
        assert (
            not self._is_post_constructed
        ), f"{self} has already been post-constructed: tried to post-construct!"
