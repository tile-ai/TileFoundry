from __future__ import annotations

import pytest
import torch

from tilefoundry.evaluator import evaluate
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Expr, Tuple, TypeInferContext, Var, VerifyError
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.target import CudaTarget
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.parser import parse_module_source
from tilefoundry.schedule import auto_dist, logical_fingerprint

SOURCE = '''from __future__ import annotations
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, tf

@module(entry="main")
class Materialize:
    @func
    def left(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return tf.add(x, x)

    @func
    def right(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return tf.add(x, x)

    @func
    def main(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        left: where(layout=(H @ cta,)) = left(x)
        right = right(x)
        return tf.add(left, right)
'''


def _walk(expr: Expr):
    if isinstance(expr, Call):
        yield expr
        for argument in expr.args:
            yield from _walk(argument)
    elif isinstance(expr, Tuple):
        for element in expr.elements:
            yield from _walk(element)


def _has_metadata(expr: Expr) -> bool:
    return bool(expr.metadata) or any(_has_metadata(child) for child in _children(expr))


def _children(expr: Expr):
    if isinstance(expr, Call):
        return expr.args
    if isinstance(expr, Tuple):
        return expr.elements
    return ()


def test_materialized_solution_is_concrete_round_trippable_and_value_preserving() -> None:
    original = parse_module_source(SOURCE)
    mesh = Mesh(Topology("cta", 8), Layout((8,), (1,)))
    result = auto_dist(
        original,
        target=CudaTarget(device="h200_sxm"),
        mesh=mesh,
    )
    solution = result.solution

    assert logical_fingerprint(solution) == logical_fingerprint(original)
    assert any(type(call.target).__name__ == "Reshard" for fn in solution.functions for call in _walk(fn.body))
    tensor_calls = [
        call
        for function in solution.functions
        for call in _walk(function.body)
        if isinstance(call.type, TensorType)
    ]
    assert tensor_calls
    assert all(isinstance(call.type.layout, ShardLayout) for call in tensor_calls)
    actual_reshards = sum(
        type(call.target).__name__ == "Reshard"
        for function in solution.functions
        for call in _walk(function.body)
    )
    assert actual_reshards == len(result.report.reshards)
    for function in solution.functions:
        assert not _has_metadata(function.body)
        assert all(not param.metadata for param in function.params)
        verify_function(function)

    printed = as_script(solution)
    reparsed = parse_module_source(printed)
    for function in reparsed.functions:
        verify_function(function)

    inputs = torch.randn(8, dtype=torch.bfloat16)
    expected = evaluate(original.entry_function(), inputs, device="cpu").data
    actual = evaluate(solution.entry_function(), inputs, device="cpu").data
    assert torch.equal(expected, actual)


def test_direct_cross_slice_binary_requires_a_reshard() -> None:
    parent = Mesh(Topology("cta", 8), Layout((8,), (1,)))
    left_type = TensorType(
        (8,),
        DType.bf16,
        ShardLayout(Layout((8,), None), (S(0),), parent[:4]),
        StorageKind.GMEM,
    )
    right_type = TensorType(
        (8,),
        DType.bf16,
        ShardLayout(Layout((8,), None), (S(0),), parent[4:]),
        StorageKind.GMEM,
    )
    left = Var(type=left_type, name="left")
    right = Var(type=right_type, name="right")
    binary = Call(
        type=left_type,
        target=Binary(kind=BinaryKind.ADD),
        args=(left, right),
    )
    with pytest.raises(VerifyError, match="different meshes"):
        TypeInferContext().type_of(binary)
