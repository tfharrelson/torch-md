from duckdb import DuckDBPyConnection
import numpy as np
from assertpy import assert_that

from torch_md.data.adapters import DuckDbAdapter
from torch_md.data.models import Calculation


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
