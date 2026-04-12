from pathlib import Path

import lightning as l
import torch
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader


def create_dataset(parquet_dir: str | Path, streaming: bool = True) -> IterableDataset:
    """
    Load a HuggingFace IterableDataset from parquet files written by a DataSink.

    The dataset is streamed from parquet shards on disk, fully decoupled
    from any database connection.
    """
    data_files = str(Path(parquet_dir) / "*.parquet")
    ds = load_dataset("parquet", data_files=data_files, streaming=streaming, split="train")
    assert isinstance(ds, IterableDataset)
    return ds


def _collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for Calculation rows.

    Handles ragged arrays (variable atom counts per molecule) by keeping
    forces/positions/masses as lists of tensors rather than stacking.
    """
    return {
        "id": [row["id"] for row in batch],
        "formula": [row["formula"] for row in batch],
        "energy": torch.tensor([row["energy"] for row in batch]),
        "forces": [torch.tensor(row["forces"]) for row in batch],
        "positions": [torch.tensor(row["positions"]) for row in batch],
        "masses": [torch.tensor(row["masses"]) for row in batch],
    }


class DFTData(l.LightningDataModule):
    """
    LightningDataModule that loads train/val/test splits from parquet directories
    via HuggingFace IterableDataset. Fully decoupled from database connections.
    """

    def __init__(
        self,
        train_dir: str | Path,
        val_dir: str | Path,
        test_dir: str | Path,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self._train_dir = train_dir
        self._val_dir = val_dir
        self._test_dir = test_dir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train_dataset = create_dataset(self._train_dir)
            self._val_dataset = create_dataset(self._val_dir)
        elif stage == "test":
            self._test_dataset = create_dataset(self._test_dir)
        elif stage == "predict":
            self._predict_dataset = create_dataset(self._test_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self._predict_dataset,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=_collate_fn,
        )
