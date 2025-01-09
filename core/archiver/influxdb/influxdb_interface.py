from typing import Optional

from docker import DockerClient
from docker.models.containers import Container
from influxdb_client import InfluxDBClient, WriteApi, QueryApi

from ..db_interface import DBInterface
from ..types import Archivable, Tags
from ...types import Config


class InfluxDBInterface(DBInterface):
    def __init__(self, influxdb_config: Config, secrets: Config) -> None:
        super().__init__(
            db_config=influxdb_config,
            secrets=secrets,
        )

        self._docker_client: Optional[DockerClient] = None
        self._idb_docker_container: Optional[Container] = None

        self._idb_client: InfluxDBClient
        self._idb_write_api: WriteApi

        self._create_connection()

    def close(self):
        self._idb_write_api.close()
        self._idb_client.close()

        if (
            self._idb_docker_container is not None
            and self._db_config["local"]["persistent"]
        ):
            self._idb_docker_container.stop()
            self._idb_docker_container.remove()

        if self._docker_client is not None:
            self._docker_client.close()

    def put(self, bucket: str, measurement: str, tags: Tags, data: Archivable) -> bool:
        if not self._does_bucket_exist(bucket):
            self._create_bucket(bucket)

        # ensures that no Tensors are passed to the InfluxDB client
        data = self._clean_data(data)

        from influxdb_client import Point

        point = Point(measurement)

        for key, value in tags.items():
            point.tag(key, value)

        for key, value in data.items():
            point.field(key, value)

        self._idb_write_api.write(bucket=bucket, record=point)

        return True

    def _create_connection(self) -> None:
        if self._db_config["local"]["enabled"]:
            self._start_influxdb_docker()

        self._idb_client = InfluxDBClient(
            url=self._db_config["url"],
            token=self._db_secrets["token"],
            org=self._db_config["org"],
        )

        from influxdb_client.client.write_api import SYNCHRONOUS

        self._idb_write_api = self._idb_client.write_api(
            write_options=SYNCHRONOUS,
        )

    def _does_bucket_exist(self, name: str) -> bool:
        buckets_api = self._idb_client.buckets_api()

        return buckets_api.find_bucket_by_name(name) is not None

    def _create_bucket(self, name: str) -> None:
        operator_client = InfluxDBClient(
            url=self._db_config["url"],
            token=self._db_secrets["operator_token"],
            org=self._db_config["org"],
        )
        buckets_api = operator_client.buckets_api()

        bucket = buckets_api.find_bucket_by_name(name)

        if bucket is None:
            buckets_api.create_bucket(bucket_name=name)

        operator_client.close()

    def _start_influxdb_docker(self) -> None:
        if self._docker_client is None:
            import docker

            self._docker_client = docker.from_env()

        running_container = self._get_running_container()

        if running_container is not None:
            self._idb_docker_container = running_container
            return

        port = self._db_config["local"]["port"]

        self._idb_docker_container = self._docker_client.containers.run(
            image=self._db_config["local"]["image"],
            name=self._db_config["local"]["container_name"],
            detach=True,
            ports={f"{port}/tcp": port},
            volumes=self._db_config["local"]["volumes"],
        )

    def _get_running_container(self) -> Optional[Container]:
        import docker.errors

        try:
            return self._docker_client.containers.get(
                self._db_config["local"]["container_name"]
            )
        except docker.errors.NotFound:
            return None
