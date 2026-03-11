from typing import Protocol

from torch_md.data.models import Calculation


class DatasetPort(Protocol):
    def load(self, data: list[Calculation]) -> None: ...
