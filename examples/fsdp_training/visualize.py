"""Export a standalone ownership explorer from the checked training HIR."""

from __future__ import annotations

import argparse
import json
from itertools import product
from math import prod
from pathlib import Path

import torch

from tilefoundry.analysis.check import check_program
from tilefoundry.ir.core import Call, Constant, Tuple, Var
from tilefoundry.ir.core.metadata import binding_name
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.shard_layout import Partial, Split, shard_layout_of, split_target_axes
from tilefoundry.ir.types.shard_projection import axis_layout, logical_projection, project_domain
from tilefoundry.ir.types.utils import leaf_span, leaves_of
from tilefoundry.runtime.tensor import logical_local_view

from .model import PARAMETER_NAMES, training_program
from .run import _transition


def describe_tensor(held):
    """Read index coverage and physical addresses without evaluating tensor data.

    Plain layouts have no inferred holder mesh. Index tags are only coordinates,
    never synthetic Partial values or a numerical check of replica equality.
    """
    shape = tuple(held.shape)
    shard = shard_layout_of(held.layout)
    result = {"shape": shape, "dtype": str(held.dtype), "storage": str(held.storage),
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
    tags = torch.arange(prod(shape)).reshape(shape)
    for coordinate in product(*(range(n) for n in mesh_shape)):
        projection = project_domain(shard.layout.shape, shape, mesh_shape, shard.attrs, coordinate)
        selected = logical_local_view(tags, held, coordinate).flatten().sort().values
        boundaries = torch.nonzero(selected[1:] != selected[:-1] + 1).flatten().tolist()
        starts, ends = [0, *(i + 1 for i in boundaries)], [*(i + 1 for i in boundaries), len(selected)]
        runs = [[int(selected[a]), int(selected[b - 1]) + 1] for a, b in zip(starts, ends, strict=True)]
        logical, logical_offset = logical_projection(shard.layout.shape, shape, mesh_shape, shard.attrs, coordinate)
        physical, physical_offset = projection.apply(shard.layout)
        physical = axis_layout(physical)
        result["participants"].append({
            "coordinate": coordinate, "shape": projection.local_shape, "runs": runs,
            "bytes": prod(projection.local_shape) * held.dtype.bit_width // 8,
            "logical_offset": int(logical_offset), "logical_strides": str(logical.strides),
            "physical_offset": int(physical_offset), "physical_map": str(physical),
        })
    return result


def trace_function(function):
    """Expand reached calls and tuple leaves by identity, retaining dependencies.

    Each call occurrence has its own bindings and memo. Loops and other region
    kinds are deliberately unsupported: the example authors both blocks.
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
                # The sequence number distinguishes repeated calls to a shared definition.
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
            raise TypeError(f"ownership explorer does not support {type(expr).__name__}")
        memo[identity] = result
        return result

    bindings = {id(param): [add(param, held, function.name, [], "Input", i)
                            for i, held in enumerate(leaves_of(param.type))]
                for param in function.params}
    outputs = walk(function.body, bindings, {}, function.name)
    return {"nodes": nodes, "views": views, "inputs": [i for ids in bindings.values() for i in ids],
            "outputs": outputs}


def training_trace(participants=4, *, replicas=2, retain=False):
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
    report.update(traces=traces, participants=participants, replicas=replicas, policy="retain" if retain else "release")
    return report


def render_html(report, *, fragment=False):
    """Embed the report without external assets, a server, or model execution."""
    template = Path(__file__).with_name("ownership.html").read_text()
    payload = json.dumps(report, separators=(",", ":")).replace("<", "\\u003c")
    content = template.replace("__OWNERSHIP_DATA__", payload)
    if fragment:
        return content
    standalone = """<style>
      :root { color-scheme: light dark; --foreground: light-dark(#222,#eee);
        --muted: light-dark(#e9e9e9,#333); --blue: light-dark(#2672bb,#75b8fa);
        --orange: light-dark(#aa671c,#e5ad64); --font-size-base: 14px; }
      body { font: 14px/1.5 system-ui,sans-serif; max-width: 900px; margin: 32px auto; padding: 0 20px; }
      h2,h3 { font-weight: 500; margin: 0; } h2 { font-size: 22px; } h3 { font-size: 15px; }
      .text-small { font-size: 12px; } .viz-row,.viz-controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
      .form-label { display: grid; flex: 1; min-width: 0; gap: 4px; }
      select,button { font: inherit; padding: 7px; } select { width: 100%; }
      .table { width: 100%; border-collapse: collapse; } th,td { text-align: left; padding: 8px; border-bottom: 1px solid var(--muted); }
      .table-responsive { overflow-x: auto; } summary { cursor: pointer; }
      @media(max-width:460px) { .viz-controls { flex-direction: column; align-items: stretch; } }
    </style>"""
    return '<!doctype html>\n<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tensor ownership</title>' + standalone + '<body>\n' + content + '\n</body></html>\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--replicas", type=int, choices=(1, 2), default=2)
    parser.add_argument("--retain", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("ownership.html"))
    args = parser.parse_args()
    if args.replicas == 2 and args.participants != 4:
        parser.error("--replicas 2 requires --participants 4; use --replicas 1 for FSDP")
    report = training_trace(args.participants, replicas=args.replicas, retain=args.retain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(report))
    print(f"Wrote {args.output}: {len(report['nodes'])} tensor values, {len(report['traces'])} parameter traces")


if __name__ == "__main__":
    main()
