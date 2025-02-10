import sys
from typing import Tuple

from ..types import CallerInfo


def get_caller_info(depth: int) -> CallerInfo:
    try:
        f = sys._getframe(depth + 1)
        if f and hasattr(f, "f_code"):
            return {
                "filename": f.f_code.co_filename,
                "lineno": f.f_lineno,
                "funcname": f.f_code.co_name,
                "modulename": f.f_globals.get("__name__", "<top>"),
            }
    except ValueError:
        pass

    return {
        "filename": "<unknown_file>",
        "lineno": -1,
        "funcname": "<unknown_function>",
        "modulename": "<unknown_module>",
    }
