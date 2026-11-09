"""Render SVG ownership diagrams from the checked training HIR."""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from math import prod
from pathlib import Path

import torch

from tilefoundry.analysis.check import check_program
from tilefoundry.core import ShardTensor, local_type
from tilefoundry.ir.expr import Call, Constant, Tuple, Var
from tilefoundry.ir.expr import binding_name
from tilefoundry.ir.core.function import Function
from tilefoundry.ir.core.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import Placement, TensorType
from tilefoundry.ir.types import ComposedLayout, Layout, flatten
from tilefoundry.ir.types import (
    Partial,
    Split,
    local_layout_and_offset,
    mesh_coords,
    shard_layout_of,
    split_target_axes,
)
from tilefoundry.ir.utils import leaf_span, leaves_of

from .model import PARAMETER_NAMES, SUPPORTED_MESHES, T, training_program
from .ownership_svg import render_svg
from .run import _transition


def describe_tensor(held):
    """Describe declared ownership and storage offsets using logical index labels.

    Plain layouts return tensor geometry with an empty participant list.
    """
    shape = tuple(held.shape)
    shard = shard_layout_of(held.layout)
    result = {"shape": shape, "dtype": held.dtype.name, "storage": str(held.storage),
              "elements": prod(shape), "mesh": [], "axes": [], "attrs": [],
              "pending": [], "participants": [], "address_map": str(held.layout)}
    if shard is None:
        return result
    mesh_shape = flatten(shard.mesh.layout).shape
    axes = [shard.mesh.names[i] if i < len(shard.mesh.names) else f"axis{i}"
            for i in range(len(mesh_shape))]
    targets = split_target_axes(shard, shape)
    result.update(mesh=mesh_shape, axes=axes, attrs=[
        f"Split({targets[i]})" if isinstance(attr, Split) else
        f"Partial({attr.reduction})" if isinstance(attr, Partial) else type(attr).__name__
        for i, attr in enumerate(shard.attrs)
    ], pending=[axes[i] for i, attr in enumerate(shard.attrs) if isinstance(attr, Partial)],
        domain=shard.layout.shape, address_map=str(shard.layout))
    tags = torch.arange(prod(shape))
    local_shape = tuple(local_type(held).shape)
    address = shard.layout.outer if isinstance(shard.layout, ComposedLayout) else shard.layout
    host = replace(shard, layout=Layout(address.shape))
    for placement in Placement.from_mesh(shard.mesh):
        coordinate = mesh_coords(shard.mesh, placement.program_ids(shard.mesh.topologies))
        logical, logical_offset = local_layout_and_offset(host, shape, coordinate)
        selected = tags.as_strided(logical.shape, logical.strides, logical_offset)
        selected = selected.flatten().sort().values
        boundaries = torch.nonzero(selected[1:] != selected[:-1] + 1).flatten().tolist()
        starts, ends = [0, *(i + 1 for i in boundaries)], [*(i + 1 for i in boundaries), len(selected)]
        runs = [[int(selected[a]), int(selected[b - 1]) + 1] for a, b in zip(starts, ends, strict=True)]
        physical, physical_offset = local_layout_and_offset(shard, shape, coordinate)
        result["participants"].append({
            "coordinate": coordinate, "shape": local_shape, "runs": runs,
            "bytes": prod(local_shape) * held.dtype.bit_width // 8,
            "logical_offset": int(logical_offset), "logical_strides": str(logical.strides),
            "physical_offset": int(physical_offset), "physical_map": str(physical),
            "index_map": {"shape": logical.shape, "strides": logical.strides},
        })
    return result


def trace_function(function):
    """Expand reached calls and tuple leaves by identity, retaining dependencies.

    Each call occurrence has fresh bindings and a memo. Scope labels include
    traversal positions to distinguish repeated calls to a shared definition.

    Raises:
        TypeError: A loop or another unsupported expression is encountered.
    """
    nodes, views, view_keys = [], [], {}

    def add(expr, held, scope, parents, op, index=0):
        if not isinstance(held, TensorType):
            return None
        key = (held.shape, held.dtype, held.storage, held.layout)
        if key not in view_keys:
            view_keys[key] = len(views)
            views.append(describe_tensor(held))
        name = getattr(expr, "name", None) or binding_name(expr) or op
        node = {"id": len(nodes), "name": name + (f"[{index}]" if index else ""),
                "scope": scope, "op": op, "parents": list(dict.fromkeys(parents)),
                "view": view_keys[key]}
        transition = _transition(expr)
        if transition:
            kind, mesh, axis = transition
            coordinates = list(product(*(range(n) for n in flatten(mesh.layout).shape)))
            groups = {}
            for coordinate in coordinates:
                group = tuple(at for i, at in enumerate(coordinate) if i != axis)
                groups.setdefault(group, []).append(coordinate)
            node["collective"] = {"kind": kind, "axis": mesh.names[axis], "groups": list(groups.values())}
        nodes.append(node)
        return node["id"]

    def walk(expr, bindings, memo, scope):
        identity = id(expr)
        if identity in bindings:
            return bindings[identity]
        if identity in memo:
            return memo[identity]
        if isinstance(expr, Constant):
            result = [None] * len(leaves_of(expr.type))
        elif isinstance(expr, Var):
            result = [add(expr, held, scope, [], "Input", i) for i, held in enumerate(leaves_of(expr.type))]
        elif isinstance(expr, Tuple):
            result = [value for element in expr.elements for value in walk(element, bindings, memo, scope)]
        elif isinstance(expr, (Call, MeshRegion)):
            args = [walk(arg, bindings, memo, scope) for arg in expr.args]
            if isinstance(expr, MeshRegion) or isinstance(expr.target, Function):
                target = expr if isinstance(expr, MeshRegion) else expr.target
                nested = {id(param): value for param, value in zip(target.params, args, strict=True)}
                name = "mesh" if isinstance(expr, MeshRegion) else target.name
                result = walk(target.body, nested, {}, f"{scope}/{name}@{len(nodes)}")
            elif isinstance(expr.target, TupleGetItem):
                start, count = leaf_span(expr.args[0].type, expr.args[1].value)
                result = args[0][start:start + count]
            else:
                parents = [value for values in args for value in values if value is not None]
                result = [add(expr, held, scope, parents, type(expr.target).__name__, i)
                          for i, held in enumerate(leaves_of(expr.type))]
        else:
            raise TypeError(f"ownership trace does not support {type(expr).__name__}")
        memo[identity] = result
        return result

    bindings = {id(param): [add(param, held, function.name, [], "Input", i)
                            for i, held in enumerate(leaves_of(param.type))]
                for param in function.params}
    outputs = walk(function.body, bindings, {}, function.name)
    return {"nodes": nodes, "views": views, "inputs": [i for ids in bindings.values() for i in ids],
            "outputs": outputs}


def training_trace(participants=16, *, replicas=4, retain=False):
    """Add model-specific parameter/output labels to the reusable HIR trace."""
    program = training_program(participants, replicas=replicas, retain=retain)
    _, function = check_program(program, program.entry_function())
    report = trace_function(function)
    nodes, views = report["nodes"], report["views"]
    count = len(PARAMETER_NAMES)
    if len(report["inputs"]) != 4 + 2 * count or len(report["outputs"]) != 2 + 3 * count:
        raise ValueError("update parameter/output labels for the changed training signature")
    traces = []
    for index, name in enumerate(PARAMETER_NAMES):
        weight = report["inputs"][4 + index]
        gathers, sources = [], {weight}
        for node in nodes:
            if node["op"] == "Reshard" and len(node["parents"]) == 1 and node["parents"][0] in sources:
                gathers.append(node["id"])
                sources.add(node["id"])
        gradient = report["outputs"][2 + index]
        chain = [gradient]
        while len(nodes[chain[-1]]["parents"]) == 1:
            parent = nodes[chain[-1]]["parents"][0]
            if views[nodes[parent]["view"]]["shape"] != views[nodes[gradient]["view"]]["shape"]:
                break
            chain.append(parent)
        stages = [{"node": weight, "label": "Parameter input"}]
        stages += [{"node": value, "label": f"Weight Reshard {i + 1}"} for i, value in enumerate(gathers)]
        stages += [{"node": value, "label": f"Gradient · {nodes[value]['name']}"} for value in reversed(chain)]
        stages += [{"node": report["outputs"][2 + 2 * count + index], "label": "Next momentum"},
                   {"node": report["outputs"][2 + count + index], "label": "Next parameter"}]
        traces.append({"name": name, "stages": stages})
    report.update(traces=traces, participants=participants, replicas=replicas,
                  batch=int(function.params[0].type.shape[0]) // T,
                  policy="retain" if retain else "release")
    return report


def stage_nodes(report, trace):
    """Name existing HIR values; gradient stages remain separate from weights."""
    nodes = report["nodes"]
    gradient = [stage["node"] for stage in trace["stages"] if stage["label"].startswith("Gradient")]
    result = {"parameter": trace["stages"][0]["node"], "partial": gradient[0],
              "gradient": gradient[-1], "momentum": trace["stages"][-2]["node"],
              "updated": trace["stages"][-1]["node"]}
    for stage in trace["stages"]:
        node = nodes[stage["node"]]
        if stage["label"].startswith("Weight Reshard"):
            result.setdefault("gather", node["id"])
        kind = node.get("collective", {}).get("kind")
        if kind in ("reduce_scatter", "all_reduce"):
            result[kind.replace("_", "-")] = node["id"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=int, choices=(1, 2, 4, 16), default=16)
    parser.add_argument("--replicas", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--retain", action="store_true")
    parser.add_argument("--tensor", choices=PARAMETER_NAMES, default="block1.q")
    parser.add_argument("--stages", nargs="+", choices=("parameter", "gather", "partial", "reduce-scatter", "all-reduce", "gradient", "momentum", "updated"))
    parser.add_argument("--value", type=int, help="render an arbitrary tensor value by its HIR trace ID")
    parser.add_argument("--list-values", action="store_true", help="list IDs for model activations and intermediates")
    parser.add_argument("--element", type=int, help="highlight this flattened global element")
    parser.add_argument("--participant", type=int, help="select this logical participant in coordinate order")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/fsdp-ownership"))
    args = parser.parse_args()
    if (args.participants, args.replicas) not in SUPPORTED_MESHES:
        parser.error(f"supported (participants, replicas): {SUPPORTED_MESHES}")
    report = training_trace(args.participants, replicas=args.replicas, retain=args.retain)
    if args.list_values:
        for node in report["nodes"]:
            print(f"{node['id']:3}  {node['name']}  {node['op']}  {node['scope']}")
        return
    if args.value is not None:
        if not 0 <= args.value < len(report["nodes"]):
            parser.error("--value is outside the tensor trace")
        selected = {f"value-{args.value}": args.value}
        name, element = "model", args.element
    else:
        trace = next(trace for trace in report["traces"] if trace["name"] == args.tensor)
        available = stage_nodes(report, trace)
        stages = args.stages or [s for s in ("partial", "reduce-scatter", "all-reduce") if s in available]
        if any(stage not in available for stage in stages):
            parser.error(f"available stages for this configuration: {tuple(available)}")
        selected = {stage: available[stage] for stage in stages}
        name = args.tensor.replace(".", "-")
        source = report["views"][report["nodes"][available["parameter"]]["view"]]["participants"]
        participant = min(1, len(source) - 1) if args.participant is None else args.participant
        if not 0 <= participant < len(source):
            parser.error("--participant is outside the logical mesh")
        element = args.element if args.element is not None else source[participant]["runs"][0][0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stage, node_id in selected.items():
        try:
            diagram = render_svg(report, node_id, element=element, participant=args.participant,
                                 title=f"{args.tensor if args.value is None else 'Model tensor'} · {stage}")
        except ValueError as error:
            parser.error(str(error))
        path = args.output_dir / f"{name}-{stage}.svg"
        path.write_text(diagram)
        print(path)


if __name__ == "__main__":
    main()
