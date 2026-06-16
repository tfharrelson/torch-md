from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from upath import UPath
from typing import ClassVar
from . import resources
import yaml
from enum import StrEnum
import os


class Env(StrEnum):
    PROD = "prod"
    DEV = "dev"


def get_env() -> Env:
    if os.environ.get("ENV") == "prod":
        return Env.PROD
    else:
        return Env.DEV


class DuckDbConfig(BaseModel):
    base_dir: UPath


class DbConfig(BaseSettings):
    duck_db: DuckDbConfig

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="DEV_DB_"
    )


class ParquetSinkConfig(BaseModel):
    base_dir: UPath


class SourceConfig(BaseModel):
    data_path: UPath
    batch_size: int


class SourcesConfig(BaseSettings):
    omol25: SourceConfig

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="DEV_SRC_"
    )

    @classmethod
    def get(cls) -> "SourcesConfig":
        if get_env() == Env.PROD:
            raise NotImplementedError("Production config not implemented yet")
        else:
            with (
                UPath(resources.__file__).parent / UPath("dev_config.yaml")
            ).open() as f:
                inputs = yaml.safe_load(f)
            return cls(**inputs)
