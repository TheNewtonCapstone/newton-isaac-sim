from enum import IntFlag, auto


class LogLevel(IntFlag):
    None_ = 0
    Debug = auto()
    Info = auto()
    Warning = auto()
    Error = auto()
    Critical = auto()


class LogOutput(IntFlag):
    None_ = 0
    File = auto()
    Console = auto()
    All = auto()
