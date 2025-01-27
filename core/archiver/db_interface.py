from abc import abstractmethod

from .types import Archivable, ArchivableConvertible, Tags
from ..types import Config


@abstractmethod
class DBInterface:
    @abstractmethod
    def __init__(self, db_config: Config, secrets: Config) -> None:
        self._db_config: Config = db_config
        self._db_secrets: Config = secrets

    @abstractmethod
    def put(
        self,
        bucket: str,
        measurement: str,
        tags: Tags,
        data: Archivable,
    ) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def _clean_data(self, data: ArchivableConvertible) -> Archivable:
        from torch import Tensor

        clean_data: Archivable = {}

        # to clean the data, we want to do a couple of things:
        # 1. If the value is None, we want to skip it
        # 2. If the value is a dictionary, flatten it recursively, for example:
        #    {"a": {"b": 1}} -> {"a.b": 1}
        # 3. If the value is a Tensor/ndarray, flatten it recursively, for example:
        #    {"a": torch.tensor(1)} -> {"a": 1}
        #    {"a": torch.tensor([1, 2])} -> {"a.x": 1, "a.y": 2}, if it's a 1D tensor of size 2 (x, y), 3 (x, y z) or 4 (x, y, z, w)
        #   If it's a 2D Tensor/ndarray, flatten it with its indices, for example:
        #    {"a": torch.tensor([[1, 2], [3, 4]])} -> {"a.0.0": 1, "a.0.1": 2, "a.1.0": 3, "a.1.1": 4}
        # 4. If the value is a list, flatten it, for example:
        #    {"a": [1, 2]} -> {"a.0": 1, "a.1": 2}

        def _idx_to_coord(idx: int, size: int) -> str:
            if size > 3:
                return str(idx)

            if idx == 0:
                return "x"
            if idx == 1:
                return "y"
            if idx == 2:
                return "z"

        for key, value in data.items():
            if value is None:
                continue

            if isinstance(value, dict):
                for sub_key, sub_value in self._clean_data(value).items():
                    clean_data[f"{key}.{sub_key}"] = sub_value
            elif isinstance(value, Tensor):
                if value.dim() == 0:
                    clean_data[key] = value.item()
                elif value.dim() == 1:
                    for i, sub_value in enumerate(value):
                        size_of_dim_zero = value.shape[0]
                        clean_data[f"{key}.{_idx_to_coord(i, size_of_dim_zero)}"] = (
                            sub_value.item()
                        )
                elif value.dim() == 2:
                    for i, row in enumerate(value):
                        for j, sub_value in enumerate(row):
                            clean_data[f"{key}.{i}.{j}"] = sub_value.item()
            elif isinstance(value, list):
                for i, sub_value in enumerate(value):
                    clean_data[f"{key}.{i}"] = sub_value
            else:
                clean_data[key] = value

        return clean_data
