import os.path
import re
from typing import Tuple, List, Optional

from core.types import Config

RUN_REGEX_STR = r"([_a-zA-Z]*)_([0-9]{3})"
CHECKPOINT_REGEX_STR = r"agent_([0-9]*).pt"
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
    checkpoint_regex = re.compile(CHECKPOINT_REGEX_STR)
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
        count: int = 0
        checkpoints: List[Config] = []

        for file in run_files:
            if file.startswith(TENSORBOARD_FILE_NAME_START):
                date = int(tensorboard_regex.match(file).group(1))
                continue

            # it's an exported file, we don't care about it here
            if (
                file.endswith(".onnx")
                or file.endswith(".yaml")
                or file.endswith(".log")
            ):
                continue

            for checkpoint in os.listdir(os.path.join(runs_folder, run_folder, file)):
                if not checkpoint.endswith(".pt"):
                    continue

                if checkpoint.startswith("best"):
                    checkpoints.append(
                        {
                            "step": -1,
                            "path": f"{runs_folder}/{run_folder}/{file}/{checkpoint}",
                        }
                    )
                    continue

                save_matches: str = checkpoint_regex.findall(checkpoint)[0]
                step: int = int(save_matches) if save_matches != "" else -1

                checkpoints.append(
                    {
                        "step": step,
                        "path": f"{runs_folder}/{run_folder}/{file}/{checkpoint}",
                    }
                )

                count += 1

        checkpoints.sort(key=lambda x: x["step"], reverse=False)

        run_settings[run_name] = {
            "id": run_id,
            "name": run_name,
            "date": date,
            "path": f"{runs_folder}/{run_folder}",
            "count": count,
            "checkpoints": checkpoints,
        }

    return dict(sorted(run_settings.items()))


def get_unused_run_id(runs_library: Config, task_name: str) -> Optional[int]:
    run_ids = [
        run["id"] for run in runs_library.values() if run["name"].startswith(task_name)
    ]

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
