from .types import LogLevel
from ..types import Config


def log_level_from_config(config: Config, key: str) -> LogLevel:
    level_str = config[key]

    if level_str == "info":
        return LogLevel.Info
    elif level_str == "debug":
        return LogLevel.Debug
    elif level_str == "warning":
        return LogLevel.Warning
    elif level_str == "error":
        return LogLevel.Error
    elif level_str == "fatal":
        return LogLevel.Critical

    return LogLevel.None_

def should_log(level: LogLevel, set_level: LogLevel) -> bool:
    return level.value >= set_level.value