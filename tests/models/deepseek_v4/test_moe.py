from __future__ import annotations

from tilefoundry import Call
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.parser import parse_schedule_func

from . import DIM, MOE_INTER, N_ACT, N_ROUTED, dsv4_moe_layer


def _walk_calls(expr):
    if isinstance(expr, Call):
        yield expr
        for argument in expr.args:
            yield from _walk_calls(argument)


def test_real_moe_layer_dimensions_and_call_boundaries():
    parsed = parse_schedule_func(dsv4_moe_layer)

    assert parsed.function.params[0].type.shape == (1, 1, DIM)
    assert parsed.function.params[2].type.shape == (N_ROUTED, DIM)
    assert parsed.function.params[4].type.shape == (N_ROUTED, MOE_INTER, DIM)
    assert parsed.function.params[10].type.shape == (MOE_INTER, DIM)
    assert parsed.function.params[0].type.dtype.value == "bf16"
    assert N_ACT == 6

    names = {call.target.name for call in _walk_calls(parsed.function.body)}
    assert {
        "pre_moe_rms_norm",
        "routed_expert",
        "shared_expert",
        "combine_expert_outputs",
    } <= names
    assert len(parsed.constraints) == 1
    assert parsed.constraints[0].storage is StorageKind.RMEM
