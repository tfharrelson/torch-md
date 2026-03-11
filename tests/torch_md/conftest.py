from collections.abc import Generator
import duckdb
from duckdb import DuckDBPyConnection
import pytest


@pytest.fixture(scope="function")
def duckdb_conn() -> Generator[DuckDBPyConnection, None, None]:
    # load an in memory connection
    with duckdb.connect(":memory:") as conn:
        yield conn
