from __future__ import annotations

import pytest

from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.parser import parse_module_source
from tilefoundry.schedule import (
    ScheduleGraphError,
    build_program_schedule_graph,
    logical_fingerprint,
)

MODULE_SOURCE = '''from __future__ import annotations
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, tf

@module(entry="main")
class Diamond:
    @func
    def leaf(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.add(x, x)

    @func
    def main(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        left: where(layout=(_, H @ cta)) = leaf(x)
        right = leaf(x)
        return tf.add(left, right)
'''


def test_parse_module_source_and_canonical_round_trip() -> None:
    module = parse_module_source(MODULE_SOURCE)
    assert module.entry == "main"
    assert [function.name for function in module.functions] == ["leaf", "main"]

    printed = as_script(module)
    reparsed = parse_module_source(printed)
    assert logical_fingerprint(reparsed) == logical_fingerprint(module)


def test_graph_expands_call_instances_with_stable_opaque_refs() -> None:
    graph = build_program_schedule_graph(parse_module_source(MODULE_SOURCE))
    root = graph.root

    assert len(root.calls) == 2
    assert len(graph.calls) == 2
    assert graph.calls[0].callee_function_id == graph.calls[1].callee_function_id
    assert graph.calls[0].call_path != graph.calls[1].call_path
    assert all(ref.function_id == graph.calls[0].callee_function_id for ref in graph.calls[0].callee_inputs)
    assert len(graph.constraints) == 1
    assert graph.constraints[0].target == graph.calls[0].result


def test_logical_fingerprint_ignores_agent_metadata() -> None:
    annotated = parse_module_source(MODULE_SOURCE)
    plain = parse_module_source(MODULE_SOURCE.replace(
        'left: where(layout=(_, H @ cta)) = leaf(x)',
        "left = leaf(x)",
    ))
    assert logical_fingerprint(annotated) == logical_fingerprint(plain)


def test_recursive_calls_are_rejected() -> None:
    tensor_type = TensorType((4,), DType.f32, None, None)
    param = Var(type=tensor_type, name="x")
    function = Function.build(
        name="recursive",
        params=(param,),
        body=None,
        return_type=tensor_type,
    )
    call = Call(type=tensor_type, target=function, args=(param,))
    object.__setattr__(function, "body", call)
    module = Module(name="recursive", functions=(function,), entry="recursive")
    with pytest.raises(ScheduleGraphError, match="recursive"):
        build_program_schedule_graph(module)
