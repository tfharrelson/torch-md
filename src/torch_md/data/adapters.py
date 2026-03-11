from tempfile import TemporaryDirectory
from typing import override
from duckdb import DuckDBPyConnection
import polars as pl
from upath import UPath

from torch_md.data.models import Calculation
from .ports import DatasetPort


class DuckDbAdapter(DatasetPort):
    def __init__(self, conn: DuckDBPyConnection):
        self._conn: DuckDBPyConnection = conn

    @override
    def load(self, data: list[Calculation]) -> None:
        tbl_result = self._conn.sql(
            "select count(*) from information_schema.tables where table_name = 'calculations'"
        ).fetchone()
        if tbl_result is None:
            raise RuntimeError("SQL command failed")
        # convert to arrow table
        with TemporaryDirectory() as d:
            t = UPath(d) / "tmp.parquet"
            pl.DataFrame(data).write_parquet(t)
            if tbl_result[0] == 0:
                _ = self._conn.sql(f"""
                    create table calculations as select * from "{t}"
                """)
            else:
                _ = self._conn.sql(
                    f"""
                    insert into calculations select * from "{t}"
                    """
                )
