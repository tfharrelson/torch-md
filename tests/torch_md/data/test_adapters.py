from pathlib import Path

from duckdb import DuckDBPyConnection
import numpy as np
import patito as pt
from assertpy import assert_that
from upath import UPath

from torch_md.data.adapters import DuckDbSink, DuckDbSource, ParquetSink
from torch_md.data.models import Calculation
from torch_md.datasets import create_dataset, DFTData


def _make_calculation_df(n: int, id_offset: int = 0) -> pt.DataFrame[Calculation]:
    """Build a pt.DataFrame[Calculation] with n rows of random test data."""
    num_atoms = 3
    return pt.DataFrame[Calculation](
        {
            "id": list(range(id_offset, id_offset + n)),
            "formula": [f"H2O_{i}" for i in range(id_offset, id_offset + n)],
            "energy": [float(i) for i in range(id_offset, id_offset + n)],
            "forces": [np.random.random(size=(num_atoms, 3)).tolist() for _ in range(n)],
            "positions": [np.random.random(size=(num_atoms, 3)).tolist() for _ in range(n)],
            "masses": [np.random.random(size=(num_atoms,)).tolist() for _ in range(n)],
        }
    )


class TestDuckDbSink:
    def test_write(self, duckdb_conn: DuckDBPyConnection):
        sink = DuckDbSink(conn=duckdb_conn)
        df = _make_calculation_df(1)
        sink.write(df)

        rows = duckdb_conn.sql("select id from calculations").fetchall()
        _ = assert_that(rows).is_length(1)
        _ = assert_that(rows[0][0]).is_equal_to(0)

    def test_write_appends(self, duckdb_conn: DuckDBPyConnection):
        sink = DuckDbSink(conn=duckdb_conn)
        sink.write(_make_calculation_df(2, id_offset=0))
        sink.write(_make_calculation_df(3, id_offset=2))

        rows = duckdb_conn.sql("select id from calculations order by id").fetchall()
        _ = assert_that(rows).is_length(5)
        _ = assert_that([r[0] for r in rows]).is_equal_to([0, 1, 2, 3, 4])


class TestParquetSink:
    def test_write_creates_parquet_files(self, tmp_path: Path):
        sink = ParquetSink(UPath(tmp_path))
        df = _make_calculation_df(5)
        sink.write(df)

        parquet_files = list(tmp_path.glob("*.parquet"))
        _ = assert_that(parquet_files).is_length(1)
        _ = assert_that(parquet_files[0].name).is_equal_to("shard_00000.parquet")

    def test_write_multiple_shards(self, tmp_path: Path):
        sink = ParquetSink(UPath(tmp_path))
        sink.write(_make_calculation_df(3, id_offset=0))
        sink.write(_make_calculation_df(3, id_offset=3))

        parquet_files = sorted(tmp_path.glob("*.parquet"))
        _ = assert_that(parquet_files).is_length(2)
        _ = assert_that(parquet_files[0].name).is_equal_to("shard_00000.parquet")
        _ = assert_that(parquet_files[1].name).is_equal_to("shard_00001.parquet")


class TestDuckDbSource:
    def test_read_batch_returns_empty_at_end(self, duckdb_conn: DuckDBPyConnection):
        sink = DuckDbSink(conn=duckdb_conn)
        sink.write(_make_calculation_df(1))

        source = DuckDbSource(duckdb_conn, "calculations", batch_size=1, val_size=0.0, test_size=0.0)

        batch1 = source.read_batch()
        _ = assert_that(len(batch1)).is_equal_to(1)
        _ = assert_that(batch1["id"].to_list()).is_equal_to([0])

        batch2 = source.read_batch()
        _ = assert_that(len(batch2)).is_equal_to(0)

    def test_reset_resets_offset(self, duckdb_conn: DuckDBPyConnection):
        sink = DuckDbSink(conn=duckdb_conn)
        sink.write(_make_calculation_df(10))

        source = DuckDbSource(duckdb_conn, "calculations", batch_size=5, val_size=0.0, test_size=0.0)

        batch1 = source.read_batch()
        _ = assert_that(len(batch1)).is_equal_to(5)

        batch2 = source.read_batch()
        _ = assert_that(len(batch2)).is_equal_to(5)

        source.reset()
        batch3 = source.read_batch()
        _ = assert_that(len(batch3)).is_equal_to(5)
        _ = assert_that(batch3["id"].to_list()[0]).is_equal_to(0)

    def test_train_val_test_split_no_overlap(self, duckdb_conn: DuckDBPyConnection):
        sink = DuckDbSink(conn=duckdb_conn)
        sink.write(_make_calculation_df(100))

        source = DuckDbSource(duckdb_conn, "calculations", batch_size=10, val_size=0.2, test_size=0.1)

        train_s, val_s, test_s = source.train_val_test_split()

        train_ids: set[int] = set()
        while True:
            batch = train_s.read_batch()
            if len(batch) == 0:
                break
            train_ids.update(batch["id"].to_list())

        val_ids: set[int] = set()
        while True:
            batch = val_s.read_batch()
            if len(batch) == 0:
                break
            val_ids.update(batch["id"].to_list())

        test_ids: set[int] = set()
        while True:
            batch = test_s.read_batch()
            if len(batch) == 0:
                break
            test_ids.update(batch["id"].to_list())

        _ = assert_that(train_ids.isdisjoint(val_ids)).is_true()
        _ = assert_that(train_ids.isdisjoint(test_ids)).is_true()
        _ = assert_that(val_ids.isdisjoint(test_ids)).is_true()

        total = len(train_ids) + len(val_ids) + len(test_ids)
        _ = assert_that(total).is_equal_to(100)


class TestCreateDataset:
    def test_create_dataset_from_parquet(self, tmp_path: Path):
        sink = ParquetSink(UPath(tmp_path))
        sink.write(_make_calculation_df(10))

        ds = create_dataset(tmp_path)

        rows = list(ds)
        _ = assert_that(len(rows)).is_equal_to(10)
        _ = assert_that(rows[0]).contains_key("id", "formula", "energy", "forces", "positions", "masses")


class TestDFTData:
    def test_dftdata_with_parquet(self, tmp_path: Path):
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        test_dir = tmp_path / "test"

        for d, n, offset in [(train_dir, 70, 0), (val_dir, 20, 70), (test_dir, 10, 90)]:
            sink = ParquetSink(UPath(d))
            sink.write(_make_calculation_df(n, id_offset=offset))

        data_module = DFTData(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            batch_size=32,
            num_workers=0,
        )

        data_module.setup("fit")

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        train_batches = list(train_loader)
        val_batches = list(val_loader)

        _ = assert_that(len(train_batches)).is_greater_than(0)
        _ = assert_that(len(val_batches)).is_greater_than(0)

        # Verify batch structure
        first_batch = train_batches[0]
        _ = assert_that(first_batch).contains_key("id", "formula", "energy", "forces", "positions", "masses")
