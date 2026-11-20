"""The ``autodiff`` command: differentiate one Module and show the result."""

from __future__ import annotations

import sys
from math import prod
from pathlib import Path
from typing import Sequence

from tilefoundry import inspection
from tilefoundry.ir import types
from tilefoundry.passes.transforms import differentiate

from .source import load_authored_ir

COLUMNS = ("saved", "shape", "bytes", "layout", "storage", "from")


def run_autodiff(
    source: str,
    wrt: str,
    recompute: Sequence[str] = (),
    out: str | None = None,
    *,
    saved: bool = False,
) -> int:
    """Write the forward and the backward of *source* as one DSL file, or its saved table."""
    module = load_authored_ir(source)
    names = tuple(name.strip() for name in wrt.split(",") if name.strip())
    forward, backward = differentiate.differentiate(module, wrt=names, recompute=tuple(recompute))
    text = saved_table(forward) if saved else inspection.as_script((forward, backward))
    if out is None:
        sys.stdout.write(text)
    else:
        Path(out).write_text(text, encoding="utf-8")
    return 0


def saved_table(forward) -> str:
    """One row per tensor the forward keeps, then the bytes its storages hold.

    A row gives the global shape, the bytes on one device, the layout, the
    storage the tensor keeps alive, and the origin. ``held`` counts each storage
    that is not a parameter once, whole, view or not; ``parameters`` counts each
    parameter storage once.
    """
    rows = [COLUMNS]
    blocks: dict[str, tuple[types.Type, str]] = {}
    for value in differentiate.saved_values(forward):
        for name, type_ in tensor_leaves(value.name, value.type):
            block = value.storage or differentiate.Saved(name, type_, value.origin)
            blocks.setdefault(block.name, (block.type, block.origin))
            rows.append(
                (name, str(tuple(type_.shape)), str(local_bytes(type_)), layout_text(type_), block.name, value.origin)
            )
    widths = [max(len(row[column]) for row in rows) for column in range(len(COLUMNS) - 1)]
    totals = {"held": 0, "parameters": 0}
    for type_, origin in blocks.values():
        whole = sum(local_bytes(leaf) for _, leaf in tensor_leaves("", type_))
        totals["parameters" if origin == "parameter" else "held"] += whole
    label = max(widths[0] + 2 + widths[1], len("parameters"))
    return "".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)) + "  " + row[-1] + "\n"
        for row in rows
    ) + "".join(f"{name.ljust(label)}  {total}\n" for name, total in totals.items())


def tensor_leaves(name: str, type_: types.Type):
    """``(name, TensorType)`` for each tensor of a saved value; a tuple's items are indexed."""
    if isinstance(type_, types.TupleType):
        for index, field in enumerate(type_.fields):
            yield from tensor_leaves(f"{name}[{index}]", field)
    else:
        yield name, type_


def local_bytes(type_: types.TensorType) -> int:
    """The bytes one device holds: the local shape, since memory is counted per device."""
    shard = types.shard_layout_of(type_.layout)
    shape = type_.shape if shard is None else types.shard_layout_local_shape(shard)
    return prod(shape) * type_.dtype.bit_width // 8


def layout_text(type_: types.TensorType) -> str:
    """Each mesh axis's attribute, a ``Split`` by the tensor axis it divides; ``-`` unsharded."""
    shard = types.shard_layout_of(type_.layout)
    if shard is None:
        return "-"
    axes = types.split_target_axes(shard, type_.shape)
    return ",".join(
        f"Split({axis})" if axis is not None else f"{type(attr).__name__}" for attr, axis in zip(shard.attrs, axes)
    )


__all__ = ["run_autodiff"]
