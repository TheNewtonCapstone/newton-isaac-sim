import os.path
import re
from typing import Tuple, List, Optional

from core.types import Config

RUN_REGEX_STR = (
    r"([a-zA-Z_]*)_([0-9]*)(?:_)?(?:rew_([-.0-9]*))?(?:_)?(?:step_([-.0-9]*))?"
)
TENSORBOARD_REGEX_STR = r"events\.out\.tfevents\.([0-9]*)"
TENSORBOARD_FILE_NAME_START = "events.out.tfevents"


def does_folder_contain_run(folder: str) -> bool:
    # we check if there's a file with that starts with "events.out.tfevents" in the folder

    if not os.path.isdir(folder):
        return False

    files = os.listdir(folder)

    for file in files:
        if file.startswith(TENSORBOARD_FILE_NAME_START):
            return True

    return False


def find_all_runs_subfolders(library_folder: str) -> list[str]:
    folders = os.listdir(library_folder)

    return [
        folder
        for folder in folders
        if does_folder_contain_run(os.path.join(library_folder, folder))
    ]


def create_runs_library(runs_folder: str) -> Config:
    os.makedirs(runs_folder, exist_ok=True)

    runs_library = build_runs_library_from_runs_folder(runs_folder)
    save_runs_library(runs_library, runs_folder)

    return runs_library


def build_runs_library_from_runs_folder(runs_folder: str) -> Config:
    run_regex = re.compile(RUN_REGEX_STR)
    tensorboard_regex = re.compile(TENSORBOARD_REGEX_STR)

    run_settings: Config = {}

    for run_folder in find_all_runs_subfolders(runs_folder):
        run_files = os.listdir(os.path.join(runs_folder, run_folder))

        if run_regex.match(run_folder) is None:
            continue

        run_matches: Tuple = run_regex.findall(run_folder)[0]
        run_id: int = int(run_matches[1])
        run_name: str = f"{run_matches[0]}_{run_id:03}"
        date: int = -1
        count: int = -1
        checkpoints: List[Config] = []

        for file in run_files:
            if file.startswith(TENSORBOARD_FILE_NAME_START):
                date = int(tensorboard_regex.match(file).group(1))
                continue

            # it's an exported file, we don't care about it here
            if file.endswith(".onnx") or file.endswith(".yaml"):
                continue

            count += 1

            if file.endswith(".zip"):
                checkpoints.append(
                    {
                        "reward": -1,
                        "step": -1,
                        "path": f"{runs_folder}/{run_folder}/{file}",
                    }
                )
                continue

            if run_regex.match(file) is None:
                continue

            save_matches: Tuple = run_regex.findall(file)[0]
            reward: float = float(save_matches[2]) if save_matches[2] != "" else -1
            step: int = int(save_matches[3]) if save_matches[3] != "" else -1

            checkpoints.append(
                {
                    "reward": reward,
                    "step": step,
                    "path": f"{runs_folder}/{run_folder}/{file}",
                }
            )

        checkpoints.sort(key=lambda x: x["reward"], reverse=False)

        run_settings[run_name] = {
            "id": run_id,
            "date": date,
            "path": f"{runs_folder}/{run_folder}",
            "count": count,
            "checkpoints": checkpoints,
        }

    return dict(sorted(run_settings.items()))


def get_unused_run_id(runs_library: Config) -> Optional[int]:
    run_ids = [run["id"] for run in runs_library.values()]

    if len(run_ids) == 0:
        return None

    run_ids.sort()

    return int(run_ids[-1]) + 1


def does_runs_library_exist(runs_folder: str) -> bool:
    return os.path.exists(os.path.join(runs_folder, "library.yaml"))


def load_runs_library(runs_folder: str) -> Config:
    from .config import load_config

    return load_config(os.path.join(runs_folder, "library.yaml"))


def save_runs_library(library: Config, runs_folder: str) -> None:
    from .config import save_config

    save_config(library, os.path.join(runs_folder, "library.yaml"))
