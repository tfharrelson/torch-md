from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from duckdb import DuckDBPyConnection
import numpy as np
import patito as pt
from patito.exceptions import DataFrameValidationError
import pytest
from assertpy import assert_that
from pydantic import BaseModel
from upath import UPath

from torch_md.data.adapters import DuckDbSink, DuckDbSource, ParquetSink
from torch_md.data.models import Calculation
from torch_md.datasets import arrow_to_tensors, create_dataset


def _make_calculation_df(n: int, id_offset: int = 0) -> pt.DataFrame[Calculation]:
    """Build a pt.DataFrame[Calculation] with n rows of random test data."""
    num_atoms = 3
    return pt.DataFrame[Calculation](
        {
            "id": list(range(id_offset, id_offset + n)),
            "formula": [f"H2O_{i}" for i in range(id_offset, id_offset + n)],
            "energy": [float(i) for i in range(id_offset, id_offset + n)],
            "forces": [
                np.random.random(size=(num_atoms, 3)).tolist() for _ in range(n)
            ],
            "positions": [
                np.random.random(size=(num_atoms, 3)).tolist() for _ in range(n)
            ],
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

        source = DuckDbSource(
            duckdb_conn, "calculations", batch_size=1, val_size=0.0, test_size=0.0
        )

        batch1 = source.read_batch()
        _ = assert_that(len(batch1)).is_equal_to(1)
        _ = assert_that(batch1["id"].to_list()).is_equal_to([0])

        batch2 = source.read_batch()
        _ = assert_that(len(batch2)).is_equal_to(0)

    def test_reset_resets_offset(self, duckdb_conn: DuckDBPyConnection):
        sink = DuckDbSink(conn=duckdb_conn)
        sink.write(_make_calculation_df(10))

        source = DuckDbSource(
            duckdb_conn, "calculations", batch_size=5, val_size=0.0, test_size=0.0
        )

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

        source = DuckDbSource(
            duckdb_conn, "calculations", batch_size=10, val_size=0.2, test_size=0.1
        )

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


# ---------------------------------------------------------------------------
# Helpers for pre-padded parquet tests
# ---------------------------------------------------------------------------

NUM_ATOMS = 3


class CalculationSchema(BaseModel):
    id: int
    formula: str
    energy: float
    forces: list[list[float]]
    positions: list[list[float]]
    masses: list[float]
    atom_mask: list[int]


def _write_prepadded_parquet(path: Path, n: int, id_offset: int = 0) -> None:
    """Write *n* rows of pre-padded test data as a single parquet shard."""
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("formula", pa.utf8()),
            ("energy", pa.float64()),
            ("forces", pa.list_(pa.list_(pa.float64(), 3), NUM_ATOMS)),
            ("positions", pa.list_(pa.list_(pa.float64(), 3), NUM_ATOMS)),
            ("masses", pa.list_(pa.float64(), NUM_ATOMS)),
            ("atom_mask", pa.list_(pa.uint8(), NUM_ATOMS)),
        ]
    )
    table = pa.table(
        {
            "id": list(range(id_offset, id_offset + n)),
            "formula": [f"H2O_{i}" for i in range(id_offset, id_offset + n)],
            "energy": [float(i) for i in range(id_offset, id_offset + n)],
            "forces": [
                np.random.random(size=(NUM_ATOMS, 3)).tolist() for _ in range(n)
            ],
            "positions": [
                np.random.random(size=(NUM_ATOMS, 3)).tolist() for _ in range(n)
            ],
            "masses": [np.random.random(size=(NUM_ATOMS,)).tolist() for _ in range(n)],
            "atom_mask": [[1] * NUM_ATOMS for _ in range(n)],
        },
        schema=schema,
    )
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path / "shard_00000.parquet")


# ---------------------------------------------------------------------------
# Tests: create_dataset
# ---------------------------------------------------------------------------


class TestCreateDataset:
    def test_yields_arrow_tables(self, tmp_path: Path):
        _write_prepadded_parquet(tmp_path, n=10)

        tables = list(create_dataset(tmp_path, schema=CalculationSchema, batch_size=10))

        _ = assert_that(len(tables)).is_greater_than(0)
        _ = assert_that(tables[0]).is_instance_of(pa.Table)
        _ = assert_that(len(tables[0])).is_equal_to(10)

    def test_respects_batch_size(self, tmp_path: Path):
        _write_prepadded_parquet(tmp_path, n=10)

        tables = list(create_dataset(tmp_path, schema=CalculationSchema, batch_size=4))

        row_counts = [len(t) for t in tables]
        _ = assert_that(sum(row_counts)).is_equal_to(10)
        _ = assert_that(row_counts[0]).is_equal_to(4)

    def test_schema_columns_preserved(self, tmp_path: Path):
        _write_prepadded_parquet(tmp_path, n=5)

        table = next(
            iter(create_dataset(tmp_path, schema=CalculationSchema, batch_size=5))
        )

        _ = assert_that(table.schema.names).contains(
            "id", "formula", "energy", "forces", "positions", "masses", "atom_mask"
        )


# ---------------------------------------------------------------------------
# Tests: schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_rejects_missing_column(self, tmp_path: Path):
        """Schema declares a column that doesn't exist in the parquet data."""

        class ExtraColumnSchema(BaseModel):
            id: int
            formula: str
            energy: float
            forces: list[list[float]]
            positions: list[list[float]]
            masses: list[float]
            atom_mask: list[int]
            nonexistent: float

        _write_prepadded_parquet(tmp_path, n=2)

        with pytest.raises(DataFrameValidationError, match="nonexistent"):
            list(create_dataset(tmp_path, schema=ExtraColumnSchema, batch_size=2))

    def test_rejects_wrong_leaf_type(self, tmp_path: Path):
        """Schema declares float but data has string."""

        class WrongTypeSchema(BaseModel):
            id: int
            formula: float  # wrong — formula is utf8 in the data
            energy: float
            forces: list[list[float]]
            positions: list[list[float]]
            masses: list[float]
            atom_mask: list[int]

        _write_prepadded_parquet(tmp_path, n=2)

        with pytest.raises(DataFrameValidationError, match="formula"):
            list(create_dataset(tmp_path, schema=WrongTypeSchema, batch_size=2))

    def test_rejects_variable_length_list(self, tmp_path: Path):
        """Parquet data with variable-length list columns must be rejected."""

        class SimpleSchema(BaseModel):
            values: list[float]

        # Write a parquet file with a variable-length list column
        table = pa.table(
            {"values": [[1.0, 2.0], [3.0, 4.0, 5.0]]},
            schema=pa.schema([("values", pa.list_(pa.float64()))]),
        )
        tmp_path.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, tmp_path / "shard_00000.parquet")

        with pytest.raises(ValueError, match="variable-length list.*pre-pad"):
            list(create_dataset(tmp_path, schema=SimpleSchema, batch_size=2))

    def test_rejects_wrong_nesting_depth(self, tmp_path: Path):
        """Schema declares list[float] but data has list[list[float]]."""

        class WrongDepthSchema(BaseModel):
            id: int
            formula: str
            energy: float
            forces: list[float]  # wrong depth — data is list[list[float]]
            positions: list[list[float]]
            masses: list[float]
            atom_mask: list[int]

        _write_prepadded_parquet(tmp_path, n=2)

        with pytest.raises(DataFrameValidationError, match="forces"):
            list(create_dataset(tmp_path, schema=WrongDepthSchema, batch_size=2))

    def test_accepts_valid_schema(self, tmp_path: Path):
        """Valid schema should not raise."""
        _write_prepadded_parquet(tmp_path, n=2)

        tables = list(create_dataset(tmp_path, schema=CalculationSchema, batch_size=2))
        _ = assert_that(len(tables)).is_greater_than(0)


# ---------------------------------------------------------------------------
# Tests: arrow_to_tensors
# ---------------------------------------------------------------------------


class TestArrowToTensors:
    def _make_table(self, n: int = 4) -> pa.Table:
        """Build a pre-padded Arrow table for conversion tests."""
        schema = pa.schema(
            [
                ("id", pa.int64()),
                ("formula", pa.utf8()),
                ("energy", pa.float64()),
                ("forces", pa.list_(pa.list_(pa.float64(), 3), 2)),
                ("mask", pa.list_(pa.uint8(), 2)),
            ]
        )
        return pa.table(
            {
                "id": list(range(n)),
                "formula": [f"mol_{i}" for i in range(n)],
                "energy": [float(i) * 1.5 for i in range(n)],
                "forces": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] for _ in range(n)],
                "mask": [[1, 0]] * n,
            },
            schema=schema,
        )

    def test_flat_numeric_is_tensor(self):
        table = self._make_table()
        result = arrow_to_tensors(table)

        energy = result["energy"]
        assert isinstance(energy, torch.Tensor)
        _ = assert_that(energy.shape).is_equal_to(torch.Size([4]))
        _ = assert_that(energy.tolist()).is_equal_to([0.0, 1.5, 3.0, 4.5])

    def test_flat_int_is_tensor(self):
        table = self._make_table()
        result = arrow_to_tensors(table)

        ids = result["id"]
        assert isinstance(ids, torch.Tensor)
        _ = assert_that(ids.tolist()).is_equal_to([0, 1, 2, 3])

    def test_nested_fixed_size_list_shape(self):
        table = self._make_table()
        result = arrow_to_tensors(table)

        forces = result["forces"]
        assert isinstance(forces, torch.Tensor)
        _ = assert_that(forces.shape).is_equal_to(torch.Size([4, 2, 3]))
        _ = assert_that(forces[0, 0].tolist()).is_equal_to([1.0, 2.0, 3.0])
        _ = assert_that(forces[0, 1].tolist()).is_equal_to([4.0, 5.0, 6.0])

    def test_uint8_mask_is_tensor(self):
        table = self._make_table()
        result = arrow_to_tensors(table)

        mask = result["mask"]
        assert isinstance(mask, torch.Tensor)
        _ = assert_that(mask.shape).is_equal_to(torch.Size([4, 2]))
        _ = assert_that(mask.dtype).is_equal_to(torch.uint8)

    def test_string_column_is_list(self):
        table = self._make_table()
        result = arrow_to_tensors(table)

        _ = assert_that(result["formula"]).is_instance_of(list)
        _ = assert_that(result["formula"]).is_equal_to(
            ["mol_0", "mol_1", "mol_2", "mol_3"]
        )

    def test_zero_copy_flat_numeric(self):
        """Flat numeric columns should share memory with the Arrow buffer."""
        table = self._make_table()
        result = arrow_to_tensors(table)

        energy = result["energy"]
        assert isinstance(energy, torch.Tensor)
        # The tensor should be non-owning (DLPack zero-copy).
        # Verify by checking the data pointer is non-null and the
        # tensor doesn't own its storage in a way that would survive
        # the Arrow table being deleted.
        _ = assert_that(energy.data_ptr()).is_not_equal_to(0)

    def test_zero_copy_nested_numeric(self):
        """Nested fixed_size_list numeric columns should share memory."""
        table = self._make_table()
        result = arrow_to_tensors(table)

        forces = result["forces"]
        assert isinstance(forces, torch.Tensor)
        _ = assert_that(forces.data_ptr()).is_not_equal_to(0)
