from pathlib import Path

import torch
from datasets import IterableDataset, load_dataset


def create_dataset(parquet_dir: str | Path, streaming: bool = True) -> IterableDataset:
    """
    Load a HuggingFace IterableDataset from parquet files written by a DataSink.

    The dataset is streamed from parquet shards on disk, fully decoupled
    from any database connection.
    """
    # TODO: just use UPath no strings or Paths
    data_files = str(Path(parquet_dir) / "*.parquet")
    ds = load_dataset(
        "parquet", data_files=data_files, streaming=streaming, split="train"
    )
    assert isinstance(ds, IterableDataset)
    return ds


def _collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for Calculation rows.

    Handles ragged arrays (variable atom counts per molecule) by keeping
    forces/positions/masses as lists of tensors rather than stacking.
    """
    # TODO: collate functions always seem so dumb - we go from columnar format
    # implement getitem to extract individual rows, then convert them back to columns
    # here. we should be able to do away with the middle man and just pass arrow
    # arrays to pytorch directly.
    return {
        # "id": [row["id"] for row in batch],
        # "formula": [row["formula"] for row in batch],
        "energy": torch.tensor([row["energy"] for row in batch]),
        "forces": [torch.tensor(row["forces"]) for row in batch],
        "positions": [torch.tensor(row["positions"]) for row in batch],
        "masses": [torch.tensor(row["masses"]) for row in batch],
    }
