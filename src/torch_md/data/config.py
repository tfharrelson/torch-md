from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from upath import UPath
from typing import ClassVar


class DuckDbConfig(BaseModel):
    base_dir: UPath


class DbConfig(BaseSettings):
    duck_db: DuckDbConfig

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="DEV_DB_"
    )


class SourceConfig(BaseModel):
    data_path: UPath
    batch_size: int


class SourcesConfig(BaseSettings):
    omol25: SourceConfig

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="DEV_SRC_"
    )
