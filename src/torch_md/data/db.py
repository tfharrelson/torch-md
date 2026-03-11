import duckdb
from upath import UPath
from .config import DuckDbConfig


def get_connection(
    db_name: str, config: DuckDbConfig, read_only: bool = True
) -> duckdb.DuckDBPyConnection:
    # TODO: think about typing the table/db names
    # TODO: work out a dev vs prod strategy
    db_subpath = UPath(f"{db_name}.duckdb")
    return duckdb.connect(str(config.base_dir / db_subpath), read_only=read_only)
