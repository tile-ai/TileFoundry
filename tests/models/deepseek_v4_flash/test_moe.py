from __future__ import annotations

from tilefoundry.ir.core import Call
from tilefoundry.schedule import build_program_schedule_graph

from . import DIM, MOE_INTER, N_ACT, N_ROUTED, dsv4_moe_module


def _walk_calls(expr):
    if isinstance(expr, Call):
        yield expr
        for argument in expr.args:
            yield from _walk_calls(argument)


def test_real_dsv4_moe_layer_has_source_dimensions_and_call_graph() -> None:
    entry = dsv4_moe_module.entry_function()
    assert entry.params[0].type.shape == (1, 1, DIM)
    assert entry.params[2].type.shape == (N_ROUTED, DIM)
    assert entry.params[4].type.shape == (N_ROUTED, MOE_INTER, DIM)
    assert entry.params[10].type.shape == (MOE_INTER, DIM)
    assert entry.params[0].type.dtype.value == "bf16"
    assert N_ACT == 6

    names = {call.target.name for call in _walk_calls(entry.body)}
    assert {
        "pre_moe_rms_norm",
        "moe_topk",
        "shared_expert",
        "combine_expert_outputs",
    } <= names

    graph = build_program_schedule_graph(dsv4_moe_module)
    assert graph.logical_fingerprint
    assert any(call.ir_call.target.name == "moe_topk" for call in graph.calls)
    assert any(call.ir_call.target.name == "moe_experts_core" for call in graph.calls)
