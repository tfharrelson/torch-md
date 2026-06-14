"""Arrow-native dataset loading with Pydantic schema validation and zero-copy tensor conversion.

The caller declares an expected schema as a Pydantic BaseModel. The loading function
validates the Arrow table schema against it and yields ``pa.Table`` batches. A separate
``arrow_to_tensors`` utility converts Arrow tables to PyTorch tensors via DLPack
(zero-copy for all numeric columns).

**Contract**: all list-typed numeric columns in the parquet files **must** use
``fixed_size_list`` types (i.e. the data must be pre-padded to rectangular shape).
Variable-length ``list`` columns are rejected at validation time.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import patito as pt
import polars as pl
import pyarrow as pa
import torch
from datasets import IterableDataset, load_dataset
from pydantic import BaseModel
from torch.utils.dlpack import from_dlpack as _from_dlpack


# ---------------------------------------------------------------------------
# Arrow type helpers
# ---------------------------------------------------------------------------


def _extract_shape_and_leaf(
    arrow_type: pa.DataType,
) -> tuple[tuple[int, ...], pa.DataType]:
    """Walk nested ``FixedSizeListType`` to return ``(shape_dims, leaf_type)``."""
    dims: list[int] = []
    t = arrow_type
    while isinstance(t, pa.FixedSizeListType):
        dims.append(t.list_size)
        t = t.value_type
    return tuple(dims), t


def _arrow_leaf_and_depth(arrow_type: pa.DataType) -> tuple[pa.DataType, int]:
    """Return ``(leaf_type, nesting_depth)`` for any Arrow type."""
    t = arrow_type
    depth = 0
    while isinstance(t, (pa.FixedSizeListType, pa.ListType)):
        t = t.value_type
        depth += 1
    return t, depth


def _is_fully_fixed(arrow_type: pa.DataType) -> bool:
    """Return ``True`` only if all list nesting uses ``fixed_size_list``."""
    t = arrow_type
    while True:
        if isinstance(t, pa.FixedSizeListType):
            t = t.value_type
        elif isinstance(t, pa.ListType):
            return False
        else:
            return True


def _array_to_list_dtype(dtype: pl.DataType) -> pl.DataType:
    """Recursively convert polars ``Array`` types to ``List`` types."""
    if isinstance(dtype, pl.Array):
        return pl.List(_array_to_list_dtype(dtype.inner))  # type: ignore[arg-type]
    return dtype


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _validate_schema(first_batch: pa.Table, pydantic_model: type[BaseModel]) -> None:
    """Validate a data sample against a caller-provided Pydantic model.

    Validation is two-phase:

    1. **Structural + type validation via patito** — a ``patito.Model`` is
       dynamically created from the Pydantic model's annotations. The first
       batch is converted to a Polars DataFrame (casting Arrow fixed-size-list
       ``Array`` types to ``List`` types for compatibility) and validated.
       Patito checks column existence and dtype compatibility.

    2. **Fixed-size-list enforcement** — all list-typed columns in the Arrow
       schema must use ``fixed_size_list``, not variable-length ``list``.
       This guarantees zero-copy conversion to tensors is possible.

    Raises ``patito.exceptions.DataFrameValidationError`` on structural /
    type mismatches and ``ValueError`` on variable-length list columns.
    """
    # --- Phase 1: patito structural + type validation ---
    patito_model: type[pt.Model] = type(
        "DynamicModel",
        (pt.Model,),
        {"__annotations__": dict(pydantic_model.__annotations__)},
    )

    pl_df = pl.from_arrow(first_batch, schema_overrides={})  # type: ignore[call-overload]
    assert isinstance(pl_df, pl.DataFrame)

    # Arrow fixed_size_list maps to polars Array, but patito expects List.
    cast_map: dict[str, pl.DataType] = {}
    for name, dtype in pl_df.collect_schema().items():
        converted = _array_to_list_dtype(dtype)
        if converted != dtype:
            cast_map[name] = converted
    if cast_map:
        pl_df = pl_df.cast(cast_map)  # type: ignore[arg-type]

    pt.DataFrame[patito_model](pl_df).set_model(patito_model).validate()  # type: ignore[type-var]

    # --- Phase 2: fixed-size-list enforcement ---
    errors: list[str] = []
    for i in range(first_batch.schema.__len__()):
        field = first_batch.schema.field(i)
        arrow_type = field.type
        _, depth = _arrow_leaf_and_depth(arrow_type)
        if depth > 0 and not _is_fully_fixed(arrow_type):
            errors.append(
                f"column '{field.name}': variable-length list type detected "
                f"({arrow_type}). All list columns must use fixed_size_list "
                f"(pre-pad your data)."
            )

    if errors:
        raise ValueError(
            "Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_BOOL_TYPES: set[pa.DataType] = {pa.bool_()}
_STRING_TYPES: set[pa.DataType] = {pa.utf8(), pa.large_utf8()}


def create_dataset[S: BaseModel](
    parquet_dir: str | Path,
    schema: type[S],
    batch_size: int,
    *,
    streaming: bool = True,
) -> Iterator[pa.Table]:
    """Load pre-padded parquet data as an iterator of Arrow tables.

    The first batch is consumed to validate the Arrow schema against the
    caller-supplied Pydantic model. If validation passes the first batch and
    all subsequent batches are yielded. If validation fails an exception is
    raised before any data is yielded.

    Parameters
    ----------
    parquet_dir:
        Directory containing ``*.parquet`` shard files.
    schema:
        A ``pydantic.BaseModel`` subclass declaring expected column names and
        base types.  List-typed columns **must** correspond to
        ``fixed_size_list`` types in the parquet files.
    batch_size:
        Number of rows per yielded ``pa.Table``.
    streaming:
        If ``True`` (default), stream from disk without loading everything
        into memory.

    Yields
    ------
    pa.Table
        Arrow tables of size ``batch_size`` (the last batch may be smaller).

    Raises
    ------
    patito.exceptions.DataFrameValidationError
        If columns are missing or have incompatible types.
    ValueError
        If any list column uses variable-length lists instead of
        ``fixed_size_list``.
    """
    data_files = str(Path(parquet_dir) / "*.parquet")
    ds = load_dataset(
        "parquet", data_files=data_files, streaming=streaming, split="train"
    )
    assert isinstance(ds, IterableDataset)
    ds = ds.with_format("arrow")

    it: Iterator[pa.Table | dict[str, list]] = ds.iter(batch_size=batch_size)  # type: ignore[assignment]

    # Consume the first batch for validation.
    first_batch: pa.Table = next(it)  # type: ignore[assignment]
    _validate_schema(first_batch, schema)

    # Yield the validated first batch, then the rest.
    yield first_batch
    for batch in it:
        yield batch  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Arrow → Tensor conversion
# ---------------------------------------------------------------------------


def _convert_column(col: pa.ChunkedArray) -> torch.Tensor | list:
    """Convert a single Arrow column to a Tensor or Python list.

    Conversion strategy by Arrow type:

    * **Primitive numeric** — ``torch.from_dlpack`` (zero-copy).
    * **``fixed_size_list<…<numeric>>``** — DLPack the flat ``.values`` buffer
      and reshape to ``[batch, d1, d2, …]`` (zero-copy).
    * **``bool`` / ``fixed_size_list<…<bool>>``** — ``to_numpy`` then
      ``torch.from_numpy`` (copy required; Arrow bit-packs booleans).
    * **``utf8`` / ``large_utf8``** — ``.to_pylist()`` → ``list[str]``.
    """
    array = col.combine_chunks()
    arrow_type = array.type

    # --- strings → Python list ---
    leaf, _ = _arrow_leaf_and_depth(arrow_type)
    if leaf in _STRING_TYPES:
        return array.to_pylist()

    # --- booleans (bit-packed, cannot DLPack) ---
    if leaf in _BOOL_TYPES:
        import numpy as np

        shape, _ = _extract_shape_and_leaf(arrow_type)
        np_arr = array.to_pandas().values  # type: ignore[union-attr]
        if shape:
            np_arr = np.stack(np_arr).reshape(len(array), *shape)
        return torch.from_numpy(np_arr)

    # --- numeric (zero-copy via DLPack) ---
    shape, _ = _extract_shape_and_leaf(arrow_type)
    if shape:
        # Walk .values through each FixedSizeListType layer to reach flat buffer
        flat: pa.Array = array
        while isinstance(flat.type, pa.FixedSizeListType):
            flat = flat.values  # type: ignore[union-attr]
        return _from_dlpack(flat).reshape(len(array), *shape)
    else:
        return _from_dlpack(array)


def arrow_to_tensors(table: pa.Table) -> dict[str, torch.Tensor | list]:
    """Convert an Arrow table to a dict of PyTorch tensors (zero-copy where possible).

    * Flat numeric columns use DLPack (true zero-copy).
    * Nested ``fixed_size_list`` numeric columns use DLPack on the flat values
      buffer and reshape (true zero-copy).
    * Boolean columns are copied (Arrow bit-packs bools; DLPack doesn't support it).
    * String columns become ``list[str]``.

    Parameters
    ----------
    table:
        An Arrow table, typically yielded by :func:`create_dataset`.

    Returns
    -------
    dict[str, torch.Tensor | list]
        Column name → tensor (or list for string columns).
    """
    return {name: _convert_column(table.column(name)) for name in table.schema.names}
