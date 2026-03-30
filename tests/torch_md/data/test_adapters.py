from duckdb import DuckDBPyConnection
import numpy as np
from assertpy import assert_that

from torch_md.data.adapters import DuckDbAdapter, DuckDbViewReader
from torch_md.data.models import Calculation
from torch_md.datasets import ReaderDataset, DFTData


class TestDuckDbAdapter:
    def test_load(self, duckdb_conn: DuckDBPyConnection):
        adapter = DuckDbAdapter(conn=duckdb_conn)
        id = 1
        calc = Calculation(
            id=id,
            formula="H2O",
            energy=12.0,
            forces=np.random.random(size=(3, 3)),
            positions=np.random.random(size=(3, 3)),
            masses=np.random.random(size=(3,)),
        )
        adapter.load([calc])
        # TODO: change this once the connection api is plumbed through the adapter
        rows = duckdb_conn.sql("select id from calculations").fetchall()
        _ = assert_that(rows).is_length(1)
        _ = assert_that(rows[0][0]).is_equal_to(id)


class TestDuckDbViewReader:
    def test_read_batch_returns_empty_list_at_end(self, duckdb_conn: DuckDBPyConnection):
        calc = Calculation(
            id=1,
            formula="H2O",
            energy=12.0,
            forces=np.random.random(size=(3, 3)),
            positions=np.random.random(size=(3, 3)),
            masses=np.random.random(size=(3,)),
        )
        adapter = DuckDbAdapter(conn=duckdb_conn)
        adapter.load([calc])

        reader = DuckDbViewReader(duckdb_conn, "calculations", batch_size=1, val_size=0.0, test_size=0.0)

        batch1 = reader.read_batch()
        _ = assert_that(len(batch1)).is_equal_to(1)
        _ = assert_that(batch1[0].id).is_equal_to(1)

        batch2 = reader.read_batch()
        _ = assert_that(batch2).is_equal_to([])

    def test_reset_resets_offset(self, duckdb_conn: DuckDBPyConnection):
        calculations = [
            Calculation(
                id=i,
                formula=f"H2O_{i}",
                energy=float(i),
                forces=np.random.random(size=(3, 3)),
                positions=np.random.random(size=(3, 3)),
                masses=np.random.random(size=(3,)),
            )
            for i in range(10)
        ]
        adapter = DuckDbAdapter(conn=duckdb_conn)
        adapter.load(calculations)

        reader = DuckDbViewReader(duckdb_conn, "calculations", batch_size=5, val_size=0.0, test_size=0.0)

        batch1 = reader.read_batch()
        _ = assert_that(len(batch1)).is_equal_to(5)
        _ = assert_that(batch1[0].id).is_equal_to(0)

        batch2 = reader.read_batch()
        _ = assert_that(len(batch2)).is_equal_to(5)
        _ = assert_that(batch2[0].id).is_equal_to(5)

        reader.reset()
        batch3 = reader.read_batch()
        _ = assert_that(len(batch3)).is_equal_to(5)
        _ = assert_that(batch3[0].id).is_equal_to(0)

    def test_train_val_test_split_no_overlap(self, duckdb_conn: DuckDBPyConnection):
        calculations = [
            Calculation(
                id=i,
                formula=f"H2O_{i}",
                energy=float(i),
                forces=np.random.random(size=(3, 3)),
                positions=np.random.random(size=(3, 3)),
                masses=np.random.random(size=(3,)),
            )
            for i in range(100)
        ]
        adapter = DuckDbAdapter(conn=duckdb_conn)
        adapter.load(calculations)

        reader = DuckDbViewReader(duckdb_conn, "calculations", batch_size=10, val_size=0.2, test_size=0.1)

        train_r, val_r, test_r = reader.train_val_test_split()

        train_ids = set()
        while batch := train_r.read_batch():
            train_ids.update(c.id for c in batch)

        val_ids = set()
        while batch := val_r.read_batch():
            val_ids.update(c.id for c in batch)

        test_ids = set()
        while batch := test_r.read_batch():
            test_ids.update(c.id for c in batch)

        _ = assert_that(train_ids.isdisjoint(val_ids)).is_true()
        _ = assert_that(train_ids.isdisjoint(test_ids)).is_true()
        _ = assert_that(val_ids.isdisjoint(test_ids)).is_true()

        total = len(train_ids) + len(val_ids) + len(test_ids)
        _ = assert_that(total).is_equal_to(100)

    def test_reader_dataset_integration(self, duckdb_conn: DuckDBPyConnection):
        calculations = [
            Calculation(
                id=i,
                formula=f"H2O_{i}",
                energy=float(i),
                forces=np.random.random(size=(3, 3)),
                positions=np.random.random(size=(3, 3)),
                masses=np.random.random(size=(3,)),
            )
            for i in range(20)
        ]
        adapter = DuckDbAdapter(conn=duckdb_conn)
        adapter.load(calculations)

        reader = DuckDbViewReader(duckdb_conn, "calculations", batch_size=5, val_size=0.0, test_size=0.0)
        dataset = ReaderDataset(reader)

        all_ids = []
        for calc in dataset:
            all_ids.append(calc.id)

        _ = assert_that(len(all_ids)).is_equal_to(20)
        _ = assert_that(set(all_ids)).is_equal_to(set(range(20)))

    def test_dftdata_with_duckdb_reader(self, duckdb_conn: DuckDBPyConnection):
        calculations = [
            Calculation(
                id=i,
                formula=f"H2O_{i}",
                energy=float(i),
                forces=np.random.random(size=(3, 3)),
                positions=np.random.random(size=(3, 3)),
                masses=np.random.random(size=(3,)),
            )
            for i in range(100)
        ]
        adapter = DuckDbAdapter(conn=duckdb_conn)
        adapter.load(calculations)

        reader = DuckDbViewReader(duckdb_conn, "calculations", batch_size=10, val_size=0.2, test_size=0.1)

        data_module = DFTData(reader, batch_size=32, num_workers=0, persistent_workers=False)

        data_module.setup("fit")

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        train_batches = list(train_loader)
        val_batches = list(val_loader)

        _ = assert_that(len(train_batches)).is_greater_than(0)
        _ = assert_that(len(val_batches)).is_greater_than(0)
