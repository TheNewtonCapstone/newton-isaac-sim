import os
from io import TextIOWrapper
from typing import Optional, Any, List, Tuple

from .types import LogLevel, LogOutput
from ..types import Config

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
        self._log_omni_buffer: List[Tuple[str, int]] = []

        if self._output & LogOutput.File and log_file_path is not None:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

            self._log_file = open(log_file_path, "w")

    def __del__(self):
        if self._log_file is not None:
            self._log_file.close()

    @property
    def _log_level(self) -> LogLevel:
        level_str = self._logger_config["level"]

        if level_str == "info":
            return LogLevel.Info
        elif level_str == "debug":
            return LogLevel.Debug
        elif level_str == "warning":
            return LogLevel.Warning
        elif level_str == "error":
            return LogLevel.Error
        elif level_str == "fatal":
            return LogLevel.Fatal
        else:
            raise ValueError(f"Unknown log level: {level_str}")

    @property
    def _output(self) -> LogOutput:
        outputs = self._logger_config["outputs"]

        if len(outputs) == 0:
            return LogOutput.None_

        output = LogOutput.None_

        for output_str in outputs:
            if output_str == "omni":
                output |= LogOutput.OmniConsole
            elif output_str == "file":
                output |= LogOutput.File
            else:
                raise ValueError(f"Unknown log output: {output_str}")

        return output

    @staticmethod
    def create(logger_config: Config, log_file_path: str) -> None:
        global _logger

        if _logger is not None:
            return

        _logger = Logger(
            logger_config=logger_config,
            log_file_path=log_file_path,
        )

    @staticmethod
    def log_level() -> LogLevel:
        global _logger

        if _logger is None:
            return LogLevel.None_

        return _logger._log_level

    @staticmethod
    def output() -> LogOutput:
        global _logger

        if _logger is None:
            return LogOutput.None_

        return _logger._output

    @staticmethod
    def flush():
        global _logger

        if _logger is None:
            return

        if _logger.output() & LogOutput.OmniConsole:
            _logger._flush_omni_buffer()

    @staticmethod
    def log(
        msg: Any,
        src_depth: int = 1,
        level: LogLevel = LogLevel.Info,
    ):
        global _logger

        if _logger is None:
            return

        if level < _logger.log_level():
            return

        _logger._print(msg, src_depth + 1)

    @staticmethod
    def info(msg: Any, src_depth: int = 1):
        Logger.log(msg, src_depth + 1, LogLevel.Info)

    @staticmethod
    def debug(msg: Any, src_depth: int = 1):
        Logger.log(msg, src_depth + 1, LogLevel.Debug)

    @staticmethod
    def warning(msg: Any, src_depth: int = 1):
        Logger.log(msg, src_depth + 1, LogLevel.Warning)

    @staticmethod
    def error(msg: Any, src_depth: int = 1):
        Logger.log(msg, src_depth + 1, LogLevel.Error)

    @staticmethod
    def fatal(msg: Any, src_depth: int = 1):
        Logger.log(msg, src_depth + 1, LogLevel.Fatal)

    def _print(self, msg: Any, src_depth: int = 1):
        if _logger.output() & LogOutput.OmniConsole:
            # If omni is not available, buffer the log message to print it later
            if "omni" not in globals():
                self._log_omni_buffer.append((msg, src_depth + 1))
            else:
                self._flush_omni_buffer()
                self._log_to_omni(str(msg), src_depth + 1)

        if _logger.output() & LogOutput.File:
            from core.utils.python import get_caller_info

            file, ln, fnc, mod = get_caller_info(src_depth + 1)
            src = f"{mod}.{fnc}():{ln}"
            msg = self._format(msg, src)

            _logger._log_to_file(msg)

    def _format(self, msg: Any, src: str) -> str:
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
            format_msg += f"[{self.log_level().name.upper()}]"

        if print_src:
            format_msg += f"[{src}]"

        format_msg += f" {msg}\n"

        return format_msg

    def _log_to_file(self, msg: str) -> None:
        if self._log_file is None:
            return

        self._log_file.write(msg)
        self._log_file.flush()  # ensure all data is written to the file

    def _log_to_omni(self, msg: str, src_depth: int) -> None:
        import omni.log as omni_logger

        if self.log_level() == LogLevel.Info:
            omni_logger.info(msg, origin_stack_depth=src_depth + 1)
        elif self.log_level() == LogLevel.Debug:
            omni_logger.verbose(msg, origin_stack_depth=src_depth + 1)
        elif self.log_level() == LogLevel.Warning:
            omni_logger.warn(msg, origin_stack_depth=src_depth + 1)
        elif self.log_level() == LogLevel.Error:
            omni_logger.error(msg, origin_stack_depth=src_depth + 1)
        elif self.log_level() == LogLevel.Fatal:
            omni_logger.fatal(msg, origin_stack_depth=src_depth + 1)

    def _flush_omni_buffer(self) -> None:
        if len(self._log_omni_buffer) == 0:
            return

        for msg, src_depth in self._log_omni_buffer:
            self._log_to_omni(msg, src_depth)

        self._log_omni_buffer.clear()
