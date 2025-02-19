import os
import pickle

from gymnasium.spaces.space import Space


def save_gymnasium_space(space: Space, path: str):
    if not path.endswith(".pkl"):
        path += ".pkl"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(space, f)


def load_gymnasium_space(path: str) -> Space:
    assert os.path.exists(path), f"Path {path} does not exist"

    with open(path, "rb") as f:
        return pickle.load(f)
