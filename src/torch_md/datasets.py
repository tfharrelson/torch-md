import lightning as l
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Iterator
from .data.ports import DataReader
from .data.models import Calculation


class ReaderDataset(IterableDataset[Calculation]):
    """
    Wraps a DataReader as a PyTorch IterableDataset.

    - Streams data from DataReader.read_batch()
    - Yields individual Calculation objects
    - PyTorch DataLoader handles final batching
    - Resets on each epoch (fresh __iter__ call)
    """

    def __init__(self, reader: DataReader):
        self.reader = reader

    def __iter__(self) -> Iterator[Calculation]:
        """
        Stream data from DataReader.

        Called once per epoch - creates fresh iterator.
        Resets DataReader state at start and end of iteration.
        Empty list signals end-of-data.
        """
        self.reader.reset()
        while True:
            batch = self.reader.read_batch()
            if not batch:
                self.reader.reset()
                break
            for item in batch:
                yield item


def _calculation_collate_fn(batch):
    """Collate function for Calculation objects - converts to dict for PyTorch"""
    return {
        "id": [c.id for c in batch],
        "formula": [c.formula for c in batch],
        "energy": torch.tensor([c.energy for c in batch]),
        "forces": torch.stack([torch.from_numpy(c.forces) for c in batch]),
        "positions": torch.stack([torch.from_numpy(c.positions) for c in batch]),
        "masses": torch.stack([torch.from_numpy(c.masses) for c in batch]),
    }


class DFTData(l.LightningDataModule):
    def __init__(
        self,
        dataset: DataReader,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        self._datareader = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers

    def prepare_data(self) -> None:
        """Currently data is already locally available."""
        pass

    def setup(self, stage: str) -> None:
        self._train_reader, self._val_reader, self._test_reader = (
            self._datareader.train_val_test_split()
        )
        self._predict_reader = self._test_reader

        if stage == "fit":
            self._train_dataset = ReaderDataset(self._train_reader)
            self._val_dataset = ReaderDataset(self._val_reader)
        elif stage == "test":
            self._test_dataset = ReaderDataset(self._test_reader)
        elif stage == "predict":
            self._predict_dataset = ReaderDataset(self._predict_reader)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            collate_fn=_calculation_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            collate_fn=_calculation_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            collate_fn=_calculation_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self._predict_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            collate_fn=_calculation_collate_fn,
        )

