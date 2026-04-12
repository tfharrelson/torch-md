from typing import Protocol

import patito as pt

from torch_md.data.models import Calculation


class DataSink(Protocol):
    """Persists batches of Calculation data to storage."""

    def write(self, data: pt.DataFrame[Calculation]) -> None: ...


class DataSource(Protocol):
    """Reads batches of Calculation data from storage."""

    def read_batch(self) -> pt.DataFrame[Calculation]: ...
    def reset(self) -> None: ...

    def train_val_test_split(
        self,
    ) -> tuple["DataSource", "DataSource", "DataSource"]: ...
