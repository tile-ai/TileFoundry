"""Shared typeinfer entry for ops tests.

Build a real ``Call(target=op, args=...)`` and run it through the
``TypeInferVisitor`` — the inferencer — instead of poking an op's private
typeinfer handler or treating the cache ``Context`` as the inferencer.
``TypeInferContext`` is only the visitor's internal cache/dispatch detail.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor


def ten(shape, dtype, *, layout=None, storage="gmem") -> TensorType:
    """Convenience TensorType builder for op input types."""
    return TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)


def _c_order(shape) -> tuple:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def mesh(layout_shape, names=None, topology="gpu") -> Mesh:
    """A ``Mesh`` with the given layout shape (C-order strides). Axis names
    default to a, b, c, … (or ``g`` for a single axis) so a test states only the
    extents instead of hand-building a ``Mesh``."""
    if names is None:
        names = ("g",) if len(layout_shape) == 1 else tuple("abcdef"[: len(layout_shape)])
    return Mesh(
        topology=topology,
        layout=Layout(shape=tuple(layout_shape), strides=_c_order(layout_shape)),
        names=tuple(names),
        topologies=(topology,),
    )


_DEFAULT = object()


def sharded(
    shape, attrs, mesh, *, cute=None, strides=_DEFAULT, dtype=DType.f32, storage="gmem"
) -> TensorType:
    """A sharded ``TensorType``. ``cute`` defaults to the tensor shape and
    ``strides`` to C-order over the cute shape, so a test states only the parts
    that matter (shape, attrs, mesh). Pass ``strides=None`` for an explicitly
    un-materialized (implicit-stride) layout."""
    cute = tuple(shape if cute is None else cute)
    if strides is _DEFAULT:
        strides = _c_order(cute)
    elif strides is not None:
        strides = tuple(strides)
    return TensorType(
        shape=tuple(shape),
        dtype=dtype,
        layout=ShardLayout(layout=Layout(shape=cute, strides=strides), attrs=tuple(attrs), mesh=mesh),
        storage=storage,
    )


def make_call(op, input_types) -> Call:
    """A ``Call`` of ``op`` over ``Var`` args carrying ``input_types``."""
    args = tuple(Var(type=t, name=f"x{i}") for i, t in enumerate(input_types))
    return Call(type=input_types[0], target=op, args=args)


def infer_call(op, *input_types):
    """Run ``op`` applied to ``input_types`` through the TypeInfer visitor."""
    return TypeInferVisitor(TypeInferContext()).visit(make_call(op, input_types))


# ─── declarative typeinfer test matrix ──────────────────────────────────────


@dataclass(frozen=True)
class ExpectedError:
    """Expected outcome: typeinfer raises ``exc`` matching ``match``. ``exc``
    defaults to ``VerifyError`` (relation-driven ops); ops that validate before
    the verify layer (e.g. ``TypeError`` on a bad axis) pass their own type."""

    match: str
    exc: type = VerifyError


@dataclass(frozen=True)
class TypeInferCase:
    """One declarative typeinfer case: apply ``op`` to ``inputs`` and expect
    either an output ``Type`` (``TensorType`` or ``TupleType``) or a raised
    error.

    An op test file declares a list of these and runs each through
    ``run_typeinfer_case``; only the per-op coverage table varies.
    """

    name: str
    op: object
    inputs: tuple[TensorType, ...]
    expected: "TensorType | TupleType | ExpectedError"


def _norm_shape(shape) -> tuple:
    """Unwrap ``Constant``-wrapped static dims to their int value so a shape of
    ``Constant(8)`` compares equal to a plain ``8`` (slice/concat emit wrapped
    dims). ``DimVar`` / dim-arithmetic ``Call`` dims (no ``value``) are left
    structural for canonical comparison."""
    return tuple(getattr(d, "value", d) for d in shape)


def assert_tensor_type(actual: TensorType, expected: TensorType) -> None:
    """Compare the typeinfer output against an expected ``TensorType`` on
    shape / dtype / storage, and on layout when the expected one pins it."""
    assert _norm_shape(actual.shape) == _norm_shape(expected.shape), (
        f"shape {actual.shape} != {expected.shape}"
    )
    assert actual.dtype == expected.dtype, f"dtype {actual.dtype} != {expected.dtype}"
    assert actual.storage == expected.storage, (
        f"storage {actual.storage} != {expected.storage}"
    )
    assert actual.layout == expected.layout, (
        f"layout {actual.layout!r} != {expected.layout!r}"
    )


def assert_type(actual, expected) -> None:
    """Compare a typeinfer output ``Type`` against ``expected``: ``TupleType``
    fieldwise, ``TensorType`` on shape/dtype/storage/layout."""
    if isinstance(expected, TupleType):
        assert isinstance(actual, TupleType), f"expected TupleType, got {actual!r}"
        assert len(actual.fields) == len(expected.fields), (
            f"tuple arity {len(actual.fields)} != {len(expected.fields)}"
        )
        for a, e in zip(actual.fields, expected.fields):
            assert_tensor_type(a, e)
        return
    assert_tensor_type(actual, expected)


def run_typeinfer_case(case: TypeInferCase) -> None:
    """Execute one ``TypeInferCase``: assert the output type or the error."""
    if isinstance(case.expected, ExpectedError):
        with pytest.raises(case.expected.exc, match=case.expected.match):
            infer_call(case.op, *case.inputs)
        return
    assert_type(infer_call(case.op, *case.inputs), case.expected)


# ─── combination builders ────────────────────────────────────────────────────
#
# The storage tiers an op test sweeps over. Op files combine these with their
# own SHAPES and LAYOUTS to enumerate input TensorTypes without hand-writing
# every combination.
STORAGES: tuple[str, ...] = ("gmem", "smem", "rmem")


def tensor_grid(shape, dtype, *, layouts=(None,), storages=STORAGES):
    """All ``TensorType`` combinations of *layouts* × *storages* for a fixed
    ``shape`` / ``dtype`` — the ``LAYOUTS × STORAGES`` axis of an op's matrix."""
    return [
        ten(shape, dtype, layout=layout, storage=storage)
        for layout in layouts
        for storage in storages
    ]
