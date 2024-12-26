import yaml
from typing import Dict, List, Any

from core.types import Settings


def load_config(config_path: str, convert_str_nones: bool = True) -> Settings:
    with open(config_path, "r") as f:
        if not convert_str_nones:
            return yaml.safe_load(f)

        return none_str_to_none(yaml.safe_load(f))


def save_config(config: Settings, config_path: str) -> None:
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def animation_configs_to_clips_settings(
        files: List[str],
) -> Dict[str, Settings]:
    clips = {}

    for file in files:
        clip = load_config(file)
        clips[clip["name"]] = clip

    return clips


def none_str_to_none(value: Settings) -> Any:
    if not isinstance(value, dict):

        if value == "None":
            return None

        return value

    for key, val in value.items():
        if isinstance(val, dict):
            value[key] = none_str_to_none(val)
            continue

        if isinstance(val, list):
            value[key] = [none_str_to_none(v) for v in val]
            continue

        if val == "None":
            value[key] = None

    return value
