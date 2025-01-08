from typing import Any, Dict, List

from numpy import ndarray
from torch import Tensor

ArchivableConvertible = Dict[
    str,
    Tensor | ndarray | List | Dict | float | str | bool | None,
]
Archivable = Dict[str, float | str | bool | None]

Tags = Dict[str, Any]
