from enum import IntFlag, auto


class LogLevel(IntFlag):
    None_ = 0
    Info = auto()
    Debug = auto()
    Warning = auto()
    Error = auto()
    Fatal = auto()


class LogOutput(IntFlag):
    None_ = 0
    OmniConsole = auto()
    File = auto()
    FileAndOmni = OmniConsole | File
    All = OmniConsole | File
