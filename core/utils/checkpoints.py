import os.path
import re
from typing import Tuple, List

import yaml
from core.types import Settings

CHECKPOINT_REGEX_STR = r"([a-zA-Z_]*)_([0-9]*)(?:_)?(?:rew_([-.0-9]*))?(?:_)?(?:step_([-.0-9]*))?"
TENSORBOARD_REGEX_STR = r"events\.out\.tfevents\.([0-9]*)"


def does_folder_contain_checkpoint(folder: str) -> bool:
    # we check if there's a file with that starts with "events.out.tfevents" in the folder

    if not os.path.isdir(folder):
        return False

    files = os.listdir(folder)

    for file in files:
        if file.startswith("events.out.tfevents"):
            return True

    return False


def find_all_checkpoint_folders(library_folder: str) -> list[str]:
    folders = os.listdir(library_folder)

    return [folder for folder in folders if does_folder_contain_checkpoint(os.path.join(library_folder, folder))]


def build_checkpoint_settings_from_library_folder(library_folder: str) -> Settings:
    checkpoint_regex = re.compile(CHECKPOINT_REGEX_STR)
    tensorboard_regex = re.compile(TENSORBOARD_REGEX_STR)

    checkpoint_settings: Settings = {}

    for checkpoint_folder in find_all_checkpoint_folders(library_folder):
        checkpoint_files = os.listdir(os.path.join(library_folder, checkpoint_folder))

        if checkpoint_regex.match(checkpoint_folder) is None:
            continue

        checkpoint_matches: Tuple = checkpoint_regex.findall(checkpoint_folder)[0]
        checkpoint_id: int = int(checkpoint_matches[1])
        checkpoint_name: str = f"{checkpoint_matches[0]}_{checkpoint_id:03}"
        date: int = -1
        count: int = -1
        saves: List[Settings] = []

        for file in checkpoint_files:
            if file.startswith("events.out.tfevents"):
                date = int(tensorboard_regex.match(file).group(1))
                continue

            count += 1

            if file.endswith(".zip"):
                saves.append({
                    "reward": -1,
                    "step": -1,
                    "path": f"{library_folder}/{checkpoint_folder}/{file}",
                })
                continue

            if checkpoint_regex.match(file) is None:
                continue

            save_matches: Tuple = checkpoint_regex.findall(file)[0]
            reward: float = float(save_matches[2]) if save_matches[2] != "" else -1
            step: int = int(save_matches[3]) if save_matches[3] != "" else -1

            saves.append({
                "reward": reward,
                "step": step,
                "path": f"{library_folder}/{checkpoint_folder}/{file}",
            })

        saves.sort(key=lambda x: x["reward"], reverse=False)

        checkpoint_settings[checkpoint_name] = {
            "id": checkpoint_id,
            "date": date,
            "path": f"{library_folder}/{checkpoint_folder}",
            "count": count,
            "saves": saves,
        }

    return dict(sorted(checkpoint_settings.items()))


def does_checkpoints_library_exist(library_folder: str) -> bool:
    return os.path.exists(os.path.join(library_folder, "library.yaml"))


def load_checkpoints_library(library_folder: str) -> Settings:
    from .config import load_config

    return load_config(os.path.join(library_folder, "library.yaml"))


def save_checkpoints_library(library: Settings, library_folder: str) -> None:
    from .config import save_config

    save_config(library, os.path.join(library_folder, "library.yaml"))
