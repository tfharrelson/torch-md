from typing import Protocol

from torch_md.data.models import Calculation


class DatasetPort(Protocol):
    def load(self, data: list[Calculation]) -> None: ...


class DataReader(Protocol):
    def read_batch(self) -> list[Calculation]: ...
    def reset(self) -> None: ...

    def train_val_test_split(
        self,
    ) -> tuple["DataReader", "DataReader", "DataReader"]: ...
