"""Run the CPU reference study: check training, analyze, compare residency."""

from __future__ import annotations

import argparse
import json
from math import prod

from tilefoundry.analysis import RegionMemoryMetadata, analyze
from tilefoundry.analysis.allocation import AllocationSummaryMetadata
from tilefoundry.analysis.check import check_program
from tilefoundry.ir.core import Call, Tuple, Var, get_metadata
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types.shard_layout import Broadcast, Partial, Split, shard_layout_of
from tilefoundry.ir.types.utils import leaf_span, leaves_of
from tilefoundry.ir.visitor import collect_exprs

from .model import training_program
from .reference import check_training


def _transition(call):
    if not isinstance(call, Call) or not isinstance(call.target, Reshard):
        return None
    before = shard_layout_of(call.args[0].type.layout)
    after = shard_layout_of(call.type.layout)
    if before is None or after is None:
        return None
    if isinstance(before.attrs[0], Split) and isinstance(after.attrs[0], Broadcast):
        return "all_gather"
    if isinstance(before.attrs[0], Partial) and isinstance(after.attrs[0], Split):
        return "reduce_scatter"
    return None


def _gathered_leaf(value, bindings=None):
    """Follow tuple/call results to a gather, without identifying by shape.

    This selects parameter gathers in this example; all gathers here are
    parameters. Allocation's accepted roots and windows still determine the
    storage identity, lifetime, and byte count.
    """
    bindings = {} if bindings is None else bindings
    expr, index = value.expr, value.index
    if isinstance(expr, Var) and id(expr) in bindings:
        arg, outer = bindings[id(expr)]
        return _gathered_leaf(SSAValue(arg, index), outer)
    if isinstance(expr, Tuple):
        for element in expr.elements:
            count = len(leaves_of(element.type))
            if index < count:
                return _gathered_leaf(SSAValue(element, index), bindings)
            index -= count
    if isinstance(expr, MeshRegion):
        nested = {id(param): (arg, bindings) for param, arg in zip(expr.params, expr.args, strict=True)}
        return _gathered_leaf(SSAValue(expr.body, index), nested)
    if isinstance(expr, Call):
        if _transition(expr) == "all_gather":
            return True
        if isinstance(expr.target, Function):
            nested = {id(param): (arg, bindings) for param, arg in zip(expr.target.params, expr.args, strict=True)}
            return _gathered_leaf(SSAValue(expr.target.body, index), nested)
        if isinstance(expr.target, TupleGetItem):
            start, _ = leaf_span(expr.args[0].type, expr.args[1].value)
            return _gathered_leaf(SSAValue(expr.args[0], start + index), bindings)
    return False


def summarize_training(result):
    """Read collective occurrences and gathered-parameter windows from HIR."""
    def occurrences(function):
        for value in collect_exprs(function.body):
            if isinstance(value, Call) and isinstance(value.target, Function):
                yield from occurrences(value.target)
            elif _transition(value):
                yield value

    participants = result.module.effective_topologies()[0].size
    counts = {"all_gather": 0, "reduce_scatter": 0}
    payload = {"all_gather": 0, "reduce_scatter": 0}
    if participants > 1:
        for call in occurrences(result.function):
            kind = _transition(call)
            counts[kind] += 1
            held = call.type if kind == "all_gather" else call.args[0].type
            payload[kind] += prod(held.shape) * held.dtype.bit_width // 8
    allocation = get_metadata(result.function, AllocationSummaryMetadata)
    memory = get_metadata(result.function, RegionMemoryMetadata)
    roots = {allocation.root(value) for value in allocation.values if _gathered_leaf(value.value)}
    windows = [(allocation.root(window.value), window) for window in memory.lifetimes]
    windows = [(root, window) for root, window in windows if root in roots]
    events = {point for _, window in windows for point in (window.defined_at, window.last_used_at)}

    def live_bytes(point):
        live = {}
        for root, window in windows:
            if window.persistent or window.defined_at <= point <= window.last_used_at:
                live[root] = max(live.get(root, 0), window.bytes)
        return sum(live.values())

    by_value = {value.value: value for value in allocation.values}
    return {
        "participants": participants,
        **counts,
        "persistent_parameter_and_momentum_bytes": sum(by_value[SSAValue(param)].bytes
                                                       for param in result.function.params[4:]),
        "full_weight_peak_bytes": max(map(live_bytes, events), default=0) if participants > 1 else None,
        "all_gather_result_bytes_per_participant": payload["all_gather"],
        "reduce_scatter_input_bytes_per_participant": payload["reduce_scatter"],
    }


def study(participants, retain):
    program = training_program(participants, retain=retain)
    checked = check_program(program, program.entry_function())
    errors = check_training(checked[1])
    result = analyze(*checked, analysis="memory")
    return {"policy": "retain" if retain else "release", "max_absolute_error": errors,
            **summarize_training(result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", nargs="+", type=int, choices=(1, 2, 4), default=(1, 2, 4))
    args = parser.parse_args()
    for participants in args.participants:
        for retain in (False, True):
            print(json.dumps(study(participants, retain)), flush=True)


if __name__ == "__main__":
    main()
