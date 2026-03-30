from tempfile import TemporaryDirectory
from typing import override
from duckdb import DuckDBPyConnection
import polars as pl
from upath import UPath
import pyarrow as pa

from torch_md.data.models import Calculation
from .ports import DataReader, DatasetPort


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
            # TODO: remove once polars figures out that upaths are paths
            pl.DataFrame(data).write_parquet(t)  # type: ignore
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


class DuckDbViewReader(DataReader):
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
    def read_batch(self) -> list[Calculation]:
        res: pa.Table = self._conn.execute(
            f"select * from {self._data_loc} limit ? offset ?",
            [self._batch_size, self._curr_offset],
        ).fetch_arrow_table()

        if len(res) == 0:
            return []

        self._curr_offset += self._batch_size
        return [Calculation.model_validate(d) for d in pl.from_arrow(res).to_dicts()]

    @override
    def reset(self) -> None:
        self._curr_offset = 0

    @override
    def train_val_test_split(self) -> tuple[DataReader, DataReader, DataReader]:
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
            DuckDbViewReader(self._conn, "train", self._batch_size, 0, 0),
            DuckDbViewReader(self._conn, "val", self._batch_size, 0, 0),
            DuckDbViewReader(self._conn, "test", self._batch_size, 0, 0),
        )
