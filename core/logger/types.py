from enum import IntFlag, auto


class LogLevel(IntFlag):
    None_ = 0
    Debug = auto()
    Info = auto()
    Warning = auto()
    Error = auto()
    Fatal = auto()


class LogOutput(IntFlag):
    None_ = 0
    CarbConsole = auto()
    File = auto()
    FileAndOmni = CarbConsole | File
    All = CarbConsole | File
