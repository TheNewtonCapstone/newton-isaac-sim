import os

import yaml
from typing import Dict, List, Any

from core.types import Config


def load_config(config_path: str, convert_str_nones: bool = True) -> Config:
    with open(config_path, "r") as f:
        if not convert_str_nones:
            return yaml.safe_load(f)

        return none_str_to_none(yaml.safe_load(f))


def save_config(config: Config, config_path: str) -> None:
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def record_configs(record_dir: str, configs: Dict[str, Config]) -> None:
    os.makedirs(record_dir, exist_ok=True)

    for name, config in configs.items():
        save_config(
            config, f"{record_dir}/{name.lower().replace(' ', '_')}_record.yaml"
        )


def animation_configs_to_clips_config(files: List[str]) -> Dict[str, Config]:
    clips = {}

    for file in files:
        clip = load_config(file)
        clips[clip["name"]] = clip

    return clips


def none_str_to_none(value: Config) -> Any:
    if isinstance(value, List):
        return [none_str_to_none(v) for v in value]

    if isinstance(value, str):
        if value == "None":
            return None

        # try to convert to float or int, if we fail, return the string
        try:
            floating = float(value)

            return int(floating) if floating.is_integer() else floating
        except ValueError:
            return value

    if isinstance(value, Dict):
        for key, val in value.items():
            value[key] = none_str_to_none(val)

    return value
