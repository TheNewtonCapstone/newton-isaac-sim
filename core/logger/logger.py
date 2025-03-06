import os
from io import TextIOWrapper
from typing import Optional, Any

from .types import LogLevel, LogOutput
from .utils import log_level_from_config, should_log
from ..types import Config, CallerInfo

_logger: Optional["Logger"] = None


class Logger:
    def __init__(
        self,
        logger_config: Config,
        log_file_path: Optional[str] = None,
    ):
        self._logger_config: Config = logger_config

        self._log_file: Optional[TextIOWrapper] = None
        self._log_file_path: Optional[str] = log_file_path

        if self._output & LogOutput.File and log_file_path is not None:
            if os.path.dirname(log_file_path) != "":
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

            self._log_file = open(log_file_path, "w")

        self._file_log_level: LogLevel = log_level_from_config(logger_config["levels"], "file")
        self._console_log_level: LogLevel = log_level_from_config(logger_config["levels"], "console")


    def __del__(self):
        if self._log_file is not None:
            self._log_file.close()

    @property
    def _output(self) -> LogOutput:
        outputs = self._logger_config["outputs"]

        if len(outputs) == 0:
            return LogOutput.None_

        output = LogOutput.None_

        for output_str in outputs:
            if output_str == "console":
                output |= LogOutput.Console
            elif output_str == "file":
                output |= LogOutput.File
            else:
                raise ValueError(f"Unknown log output: {output_str}")

        return output

    @staticmethod
    def create(logger_config: Config, log_file_path: Optional[str] = None) -> None:
        global _logger

        if _logger is not None:
            return

        _logger = Logger(
            logger_config=logger_config,
            log_file_path=log_file_path,
        )

        _logger.info("Logger initialized!")

    @staticmethod
    def set_log_file_path(log_file_path: str) -> None:
        global _logger

        _logger._log_file_path = log_file_path

        if not (_logger._output & LogOutput.File):
            return

        if _logger._log_file is not None:
            _logger._log_file.close()

        if os.path.dirname(log_file_path) != "":
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        _logger._log_file = open(log_file_path, "w")

    @staticmethod
    def output() -> LogOutput:
        global _logger

        if _logger is None:
            return LogOutput.None_

        return _logger._output

    @staticmethod
    def log(
        msg: Any,
        src_depth: int = 1,
        level: LogLevel = LogLevel.Info,
    ):
        global _logger

        if _logger is None:
            return

        _logger._log(msg, level, src_depth + 1)

    @staticmethod
    def info(msg: Any, src_depth: int = 0):
        Logger.log(msg, src_depth + 1, LogLevel.Info)

    @staticmethod
    def debug(msg: Any, src_depth: int = 0):
        Logger.log(msg, src_depth + 1, LogLevel.Debug)

    @staticmethod
    def warning(msg: Any, src_depth: int = 0):
        Logger.log(msg, src_depth + 1, LogLevel.Warning)

    @staticmethod
    def error(msg: Any, src_depth: int = 0):
        Logger.log(msg, src_depth + 1, LogLevel.Error)

    @staticmethod
    def fatal(msg: Any, src_depth: int = 0):
        Logger.log(msg, src_depth + 1, LogLevel.Critical)

    def _log(self, msg: Any, level: LogLevel, src_depth: int = 1):
        from ..utils.python import get_caller_info

        caller_info = get_caller_info(src_depth + 1)
        src = f"{caller_info['modulename']}.{caller_info['funcname']}():{caller_info['lineno']}"
        msg = self._format(msg, src, level)

        if _logger.output() & LogOutput.Console and should_log(level, self._console_log_level):
            self._log_to_console(msg)

        if _logger.output() & LogOutput.File and should_log(level, self._file_log_level):
            _logger._log_to_file(msg)

    def _format(self, msg: Any, src: str, level: LogLevel) -> str:
        print_date = self._logger_config["format"]["date"]
        print_time = self._logger_config["format"]["time"]
        print_level = self._logger_config["format"]["level"]
        print_src = self._logger_config["format"]["source"]

        format_msg = ""

        if print_date:
            from datetime import datetime

            date = datetime.now().strftime("%Y-%m-%d")
            format_msg += f"[{date}]"

        if print_time:
            from datetime import datetime

            time = datetime.now().strftime("%H:%M:%S/%f")
            format_msg += f"[{time}]"

        if print_level:
            format_msg += f"[{level.name.upper()}]"

        if print_src:
            format_msg += f"[{src}]"

        format_msg += f" {msg}\n"

        return format_msg

    def _log_to_file(self, msg: str) -> None:
        if self._log_file is None:
            return

        self._log_file.write(msg)
        self._log_file.flush()  # ensure all data is written to the file

    def _log_to_console(self, msg: str) -> None: # noqa
        print(msg)
