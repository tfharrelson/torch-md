from typing import override

from duckdb import DuckDBPyConnection
import patito as pt
import polars as pl
from upath import UPath

from torch_md.data.models import Calculation
from .ports import DataSink, DataSource


class DuckDbSink(DataSink):
    """Persists Calculation DataFrames into a DuckDB table."""

    def __init__(self, conn: DuckDBPyConnection):
        self._conn: DuckDBPyConnection = conn

    @override
    def write(self, data: pt.DataFrame[Calculation]) -> None:
        tbl_result = self._conn.sql(
            "select count(*) from information_schema.tables where table_name = 'calculations'"
        ).fetchone()
        if tbl_result is None:
            raise RuntimeError("SQL command failed")

        arrow_table = data.to_arrow()
        if tbl_result[0] == 0:
            self._conn.sql("create table calculations as select * from arrow_table")
        else:
            self._conn.sql("insert into calculations select * from arrow_table")


class ParquetSink(DataSink):
    """Persists Calculation DataFrames as parquet shard files in a directory."""

    def __init__(self, base_dir: UPath):
        self._base_dir = base_dir
        self._shard_idx = 0
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @override
    def write(self, data: pt.DataFrame[Calculation]) -> None:
        path = self._base_dir / f"shard_{self._shard_idx:05d}.parquet"
        # patito DataFrame extends polars DataFrame, so write_parquet is available
        data.write_parquet(path)  # type: ignore[arg-type]
        self._shard_idx += 1


class DuckDbSource(DataSource):
    """Reads Calculation data from DuckDB views/tables in batches."""

    def __init__(
        self,
        conn: DuckDBPyConnection,
        data_loc: str,
        batch_size: int,
        val_size: float = 0.2,
        test_size: float = 0.1,
    ):
        self._conn = conn
        self._batch_size = batch_size
        self._curr_offset = 0
        self._data_loc = data_loc
        self._val_size = val_size
        self._test_size = test_size

    @override
    def read_batch(self) -> pt.DataFrame[Calculation]:
        arrow_table = self._conn.execute(
            f"select * from {self._data_loc} limit ? offset ?",
            [self._batch_size, self._curr_offset],
        ).fetch_arrow_table()

        self._curr_offset += self._batch_size
        return pt.DataFrame[Calculation](pl.from_arrow(arrow_table))

    @override
    def reset(self) -> None:
        self._curr_offset = 0

    @override
    def train_val_test_split(self) -> tuple[DataSource, DataSource, DataSource]:
        count_result = self._conn.sql("select count(*) from calculations").fetchone()
        if count_result is None:
            raise RuntimeError("Count of records in calculations table is None")
        count = count_result[0]

        n_test = int(count * self._test_size)
        n_val = int(count * self._val_size)
        n_train = count - n_test - n_val

        self._conn.execute(f"""
            CREATE OR REPLACE VIEW train AS
            SELECT * FROM calculations ORDER BY id LIMIT {n_train} OFFSET 0
        """)
        self._conn.execute(f"""
            CREATE OR REPLACE VIEW val AS
            SELECT * FROM calculations ORDER BY id LIMIT {n_val} OFFSET {n_train}
        """)
        self._conn.execute(f"""
            CREATE OR REPLACE VIEW test AS
            SELECT * FROM calculations ORDER BY id LIMIT {n_test} OFFSET {n_train + n_val}
        """)

        return (
            DuckDbSource(self._conn, "train", self._batch_size, 0, 0),
            DuckDbSource(self._conn, "val", self._batch_size, 0, 0),
            DuckDbSource(self._conn, "test", self._batch_size, 0, 0),
        )
