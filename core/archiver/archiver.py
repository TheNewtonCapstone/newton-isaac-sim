from typing import Optional

from .db_interface import DBInterface
from .types import ArchivableConvertible, Tags
from ..logger import Logger
from ..types import Config
from ..universe import Universe

_archiver: Optional["Archiver"] = None


class Archiver:
    def __init__(self, universe: Universe, db_config: Config, secrets: Config):
        self._universe: Universe = universe

        self.db_config: Config = db_config
        self.db_secrets: Config = secrets

        self._enabled: bool = self.db_config["enabled"]

        self._db_interface: DBInterface

        self._connect_db()

    def __del__(self):
        self._db_interface.close()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def disabled(self) -> bool:
        return not self._enabled

    @staticmethod
    def create(universe: Universe, db_config: Config, secrets: Config) -> None:
        global _archiver

        if _archiver is not None:
            return

        _archiver = Archiver(
            universe=universe,
            db_config=db_config,
            secrets=secrets,
        )

    @staticmethod
    def put(
        measurement: str,
        data: ArchivableConvertible,
        bucket: str = "main",
    ) -> bool:
        global _archiver

        if _archiver is None:
            return False

        return _archiver._put(
            measurement=measurement,
            data=data,
            bucket=bucket,
        )

    def _connect_db(self) -> None:
        if self.disabled:
            return None

        if self.db_config["type"] == "influxdb":
            Logger.info("Connecting to InfluxDB...")

            from .influxdb import InfluxDBInterface

            self._db_interface = InfluxDBInterface(
                influxdb_config=self.db_config["influxdb"],
                secrets=self.db_secrets["influxdb"],
            )
        else:
            raise NotImplemented(
                f"Database type {self.db_config['type']} is not supported."
            )

    def _put(self, measurement: str, data: ArchivableConvertible, bucket: str) -> bool:
        if self.disabled:
            return False

        tags: Tags = {
            "source": "sim",
            "mode": self._universe.mode,
            "run": self._universe.run_name,
        }

        return self._db_interface.put(
            bucket=bucket,
            measurement=measurement,
            tags=tags,
            data=data,
        )
