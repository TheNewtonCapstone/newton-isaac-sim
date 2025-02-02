import sys
from typing import Tuple


def get_caller_info(depth: int) -> Tuple[str, int, str, str]:
    try:
        f = sys._getframe(depth + 1)
        if f and hasattr(f, "f_code"):
            return (
                f.f_code.co_filename,
                f.f_lineno,
                f.f_code.co_name,
                f.f_globals.get("__name__", "<top>"),
            )
    except ValueError:
        pass
    return "(unknown file)", 0, "(unknown function)", "(unknown module)"
