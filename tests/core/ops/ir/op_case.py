"""One op case shares its Call and inferred type across the stated checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import pytest
import torch

from tests.core._corpus import Problems
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType, TensorType, Topology, TupleType, make_tensor_type
from tilefoundry.ir.types.layout import Layout
from tilefoundry.ir.types.shard_layout import ShardLayout, Split, shard_layout_local_shape
from tilefoundry.visitor_registry.contexts import CostContext, TrafficBytes, TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor
from tilefoundry.visitor_registry.visitors import CostEvaluator


@dataclass(frozen=True)
class ExpectedError:
    match: str
    exc: type = VerifyError


@dataclass(frozen=True)
class Cost:
    flops: Mapping[DType, int] = field(default_factory=dict)
    ops: Mapping[str, int] = field(default_factory=dict)
    traffic: tuple[TrafficBytes, ...] = ()
    topology_level: str | None = None
    topologies: tuple[Topology, ...] = ()


@dataclass(frozen=True)
class OpCase:
    name: str
    op: object
    inputs: tuple[TensorType | torch.Tensor, ...]
    type: TensorType | TupleType | None = None
    error: ExpectedError | None = None
    value: torch.Tensor | tuple | None = None
    cost: Cost | None = None
    target: object | None = None
    storages: tuple[str, ...] = ()
    atol: float = 1e-5
    rtol: float = 1e-5

    @property
    def id(self) -> str:
        return self.name


def make_call(op, input_types) -> Call:
    """Build a Call over typed variables, with a placeholder for nullary ops."""
    args = tuple(Var(type=t, name=f"x{i}") for i, t in enumerate(input_types))
    return Call(
        type=input_types[0] if input_types else TensorType.umat_scalar(), target=op, args=args
    )


def _infer(call, target=None):
    if target is None:
        return TypeInferVisitor().visit(call, TypeInferContext())
    function = Function.build(name="case", params=call.args, body=call, return_type=call.type)
    module = Module(name="Case", functions=(function,), entry="case", target=target)
    return TypeInferVisitor().visit(call, TypeInferContext(module=module, function=function))


def infer_call(op, *input_types, target=None):
    """Infer one op in a Module scope only when the caller supplies a target."""
    return _infer(make_call(op, input_types), target=target)


_DTYPE_OF = {
    torch.float32: DType.f32,
    torch.float16: DType.f16,
    torch.bfloat16: DType.bf16,
    torch.float8_e4m3fn: DType.fp8e4m3,
    torch.int32: DType.i32,
    torch.int64: DType.i64,
    torch.bool: DType.bool,
}


def tensor_type_of(t: torch.Tensor, storage: str = "gmem") -> TensorType:
    return make_tensor_type(tuple(t.shape), _DTYPE_OF[t.dtype], storage)


def _input_types(case):
    storages = case.storages or ("gmem",) * len(case.inputs)
    return tuple(
        tensor_type_of(value, storage) if isinstance(value, torch.Tensor) else value
        for value, storage in zip(case.inputs, storages)
    )


def _evaluate(call, inputs):
    function = Function.build(name="eval_case", params=call.args, body=call, return_type=call.type)
    return evaluate(function, *inputs)


def _cost(call, expected, input_types):
    context = CostContext(
        selected_types={id(arg): type_ for arg, type_ in zip(call.args, input_types)},
        selected_output_type=call.type,
        topology_level=expected.topology_level,
        topologies=expected.topologies,
    )
    return CostEvaluator().visit_Call(call, context)


def run_op_case(case: OpCase, problems: Problems) -> None:
    types = problems.run("input types", _input_types, case)
    call = problems.run("Call", make_call, case.op, types)
    if case.error is not None:

        def refused(call):
            with pytest.raises(case.error.exc, match=case.error.match):
                _infer(call, target=case.target)

        problems.run("type refusal", refused, call)
        return
    inferred = problems.run("infer", _infer, call, target=case.target)
    typed = problems.run("typed Call", lambda c, t: replace(c, type=t), call, inferred)
    if case.type is not None:
        problems.run("type", assert_type, inferred, case.type)
    if case.value is not None:
        actual = problems.run("evaluate", _evaluate, typed, case.inputs)
        problems.run("value", _assert_value, actual, case.value, atol=case.atol, rtol=case.rtol)
    if case.cost is not None:
        cost = problems.run("cost", _cost, typed, case.cost, types)
        problems.expect("flops", lambda c: c.flops == case.cost.flops, cost)
        problems.expect("ops", lambda c: c.ops == case.cost.ops, cost)
        problems.expect("traffic", lambda c: c.traffic == case.cost.traffic, cost)


def check_op_case(case: OpCase) -> None:
    problems = Problems(case)
    run_op_case(case, problems)
    problems.dump()


def op_case_test(cases: Sequence[OpCase]):
    @pytest.mark.parametrize("case", cases, ids=lambda case: case.name)
    def test_op_case(case):
        check_op_case(case)

    return test_op_case


def raw_shard_tensor_type(
    shape, layout_shape, strides, attrs, mesh, *, dtype=DType.f32, storage="gmem"
) -> TensorType:
    """Raw shard tensor type.

    A sharded ``TensorType`` built directly from an explicit (possibly
    non-canonical) ``layout_shape``/``strides``, bypassing
    ``make_shard_tensor_type``'s canonicalization. Only for the handful of
    tests exercising a layout shape ``make_shard_tensor_type`` cannot itself
    produce (an implicit ``strides=None`` layout, or a stride pattern that
    isn't plain C-order).
    """
    return TensorType(
        shape=shape,
        dtype=dtype,
        storage=storage,
        layout=ShardLayout(
            layout=Layout(shape=layout_shape, strides=strides), attrs=attrs, mesh=mesh
        ),
    )


def split_local_extents(ty) -> list:
    """`shard_layout_local_shape` at every `Split`-bound layout dim of *ty*'s layout.

    Every entry must be 1 on a canonical layout
    ([shard §7.1.1](docs/spec/shard.md#711-layoutshape)).
    """
    local = shard_layout_local_shape(ty.layout)
    return [local[a.axis] for a in ty.layout.attrs if isinstance(a, Split)]


def _norm_shape(shape) -> tuple:
    """Norm shape.

    Unwrap ``Constant``-wrapped static dims to their int value so a shape of
    ``Constant(8)`` compares equal to a plain ``8`` (slice/concat emit wrapped
    dims). ``DimVar`` / dim-arithmetic ``Call`` dims (no ``value``) are left
    structural for canonical comparison.
    """
    return tuple(getattr(d, "value", d) for d in shape)


def assert_tensor_type(actual: TensorType, expected: TensorType) -> None:
    """Compare the typeinfer output against an expected ``TensorType`` on shape / dtype / storage.

    Compare the typeinfer output against an expected ``TensorType`` on
    shape / dtype / storage, and on layout when the expected one pins it.
    """
    assert _norm_shape(actual.shape) == _norm_shape(expected.shape), (
        f"shape {actual.shape} != {expected.shape}"
    )
    assert actual.dtype == expected.dtype, f"dtype {actual.dtype} != {expected.dtype}"
    assert actual.storage == expected.storage, f"storage {actual.storage} != {expected.storage}"
    assert actual.layout == expected.layout, f"layout {actual.layout!r} != {expected.layout!r}"


def assert_type(actual, expected) -> None:
    """Compare a typeinfer output ``Type`` against ``expected``.

    Compare a typeinfer output ``Type`` against ``expected``: ``TupleType``
    fieldwise, ``TensorType`` on shape/dtype/storage/layout.
    """
    if isinstance(expected, TupleType):
        assert isinstance(actual, TupleType), f"expected TupleType, got {actual!r}"
        assert len(actual.fields) == len(expected.fields), (
            f"tuple arity {len(actual.fields)} != {len(expected.fields)}"
        )
        for a, e in zip(actual.fields, expected.fields):
            assert_tensor_type(a, e)
        return
    assert_tensor_type(actual, expected)


def _assert_value(actual, expected, *, atol: float, rtol: float) -> None:
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for actual_field, expected_field in zip(actual, expected):
            _assert_value(actual_field, expected_field, atol=atol, rtol=rtol)
        return
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
