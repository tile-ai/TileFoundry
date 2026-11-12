"""Stress virtual PP/CP/EP grids with HIR contracts and explicitly inferred timing."""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import gzip
import heapq
import json
from math import log2
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.distributed.pipelining import Schedule1F1B

from tilefoundry.ir import types
from tilefoundry.ir.hir.sharding import Collective, Exchange

from config import FULL
from model import collective_function
from timeline import duration, read_rank


class Calibration:
    def __init__(self, path):
        self.data = json.loads(path.read_text())
        self.samples = defaultdict(list)
        for row in self.data["records"]:
            self.samples[row["kind"], row["size"]].append((row["bytes"], row["seconds"]))
        for rows in self.samples.values():
            rows.sort()

    def seconds(self, kind, size, nbytes):
        if size == 1 or nbytes == 0:
            return 0.0
        measured_size = min((2, 4, 8), key=lambda n: abs(log2(size / n)))
        rows = self.samples[kind, measured_size]
        left, right = rows[0], rows[1]
        for a, b in zip(rows, rows[1:]):
            left, right = a, b
            if nbytes <= b[0]:
                break
        slope = max(0, (right[1] - left[1]) / (right[0] - left[0]))
        base = max(rows[0][1] / 2, left[1] - slope * left[0])
        rounds = (log2(size) / log2(measured_size) if kind in {"broadcast", "reduce"}
                  else (size - 1) / (measured_size - 1))
        traffic = rounds if kind in {"all_gather", "reduce_scatter"} else 1
        return base * rounds + slope * nbytes * traffic


class Graph:
    def __init__(self):
        self.nodes = {}

    def add(self, key, duration, deps=(), resources=(), **info):
        if key in self.nodes:
            raise ValueError(f"duplicate event: {key}")
        self.nodes[key] = dict(duration=duration, deps=set(deps), resources=set(resources), **info)
        return key

    def run(self):
        children, pending, ready, finish, resources = defaultdict(list), {}, [], {}, defaultdict(float)
        for key, node in self.nodes.items():
            pending[key] = len(node["deps"])
            for parent in node["deps"]:
                if parent not in self.nodes:
                    raise ValueError(f"missing dependency: {parent}")
                children[parent].append(key)
            if not pending[key]:
                heapq.heappush(ready, (0.0, node.get("priority", 0), key))
        events = []
        while ready:
            earliest, priority, key = heapq.heappop(ready)
            node = self.nodes[key]
            start = max([earliest, *(resources[r] for r in node["resources"])])
            if ready and start > ready[0][0]:
                heapq.heappush(ready, (start, priority, key))
                continue
            end = start + node["duration"]
            finish[key] = end
            for resource in node["resources"]:
                resources[resource] = end
            events.append(dict(name=node["name"], ph="X", pid=node["stage"], tid=node["lane"],
                               ts=start * 1e6, dur=node["duration"] * 1e6,
                               args=dict(step=node["step"], microbatch=node.get("microbatch"))))
            for child in children[key]:
                pending[child] -= 1
                if not pending[child]:
                    heapq.heappush(ready, (max(finish[p] for p in self.nodes[child]["deps"]),
                                           self.nodes[child].get("priority", 0), child))
        if len(finish) != len(self.nodes):
            raise ValueError(f"dependency cycle: {len(self.nodes) - len(finish)} events blocked")
        return max(finish.values()), events


def schedule_order(stages, microbatches):
    """Reuse PyTorch's action order, normalizing its diagnostic microbatch labels."""
    schedule = SimpleNamespace(_num_stages=stages, _n_microbatches=microbatches)
    raw = Schedule1F1B._get_pipeline_order(schedule)
    result = {}
    for stage, actions in raw.items():
        counts, ordered = defaultdict(int), []
        for action in actions:
            if action is None:
                continue
            kind = {"FORWARD": "F", "FULL_BACKWARD": "B"}.get(action.computation_type.name)
            if kind in {"F", "B"}:
                ordered.append((kind, counts[kind]))
                counts[kind] += 1
        if dict(counts) != {"F": microbatches, "B": microbatches}:
            raise ValueError(f"unsupported installed PyTorch schedule: {counts}")
        result[stage] = ordered
    return result


def compute_calibration(folder):
    manifests = sorted(folder.glob("rank*.json"))
    if not manifests:
        raise ValueError("the numerical gate requires completed rank manifests")
    values, baseline = defaultdict(list), None
    for path in manifests:
        rank = int(path.stem.removeprefix("rank"))
        data = read_rank(folder, rank)
        manifest = data["manifest"]
        if manifest["status"] != "passed" or not manifest.get("packed"):
            raise ValueError("calibration requires a passed packed CP/EP training run")
        baseline = manifest
        measured = manifest["profile"]["profiled_steps"]
        for phase in ("forward", "backward", "optimizer"):
            seconds = duration(data["intervals"].get(phase, ())) / 1e6
            count = measured if phase == "optimizer" else measured * manifest["microbatches"]
            role = "last" if manifest["stage"] == manifest["pp"] - 1 else "body"
            values[role, phase].append(seconds / count)
    if len(manifests) != baseline["pp"] * baseline["dp"]:
        raise ValueError("incomplete numerical gate")
    return baseline, {f"{role}/{phase}": max(times) for (role, phase), times in values.items()}


def stress_grid(grid, args, baseline, compute, native, remote):
    pp, cp, ep, data = grid
    dense, world = cp * ep * data, pp * cp * ep * data
    shape = replace(FULL, layers=args.layers)
    if shape.layers % pp or shape.experts % ep or args.microbatches < pp:
        raise ValueError("layers must divide PP, experts must divide EP, microbatches must cover PP")
    tokens, local = args.tokens_per_cp * cp, args.tokens_per_cp
    mesh = types.Mesh((types.Topology("gpu", world),), types.Layout((pp, data, ep, cp)),
                      ("pp", "data", "ep", "cp"))
    contracts, communication = [], defaultdict(list)

    def contract(kind, members, payload, *, counts=None, phase, active=None, dtype=torch.float32):
        ranks = tuple(p.ids["gpu"] for p in types.Placement.from_mesh(members))
        context = f"virtual/{phase}/" + ",".join(map(str, ranks))
        if counts is None:
            op = Collective(participants=members, context=context, kind=kind,
                            root=0 if kind in {"reduce", "broadcast"} else None,
                            reduction="sum" if kind in {"reduce", "reduce_scatter"} else None)
        else:
            op = Exchange(participants=members, context=context, counts=counts)
        inferred = collective_function(op, (payload,), dtype)
        assert len(set(ranks)) == len(ranks) and all(0 <= rank < world for rank in ranks)
        nodes = {rank // args.gpus_per_node for rank in active or ranks}
        kind_key = "exchange" if counts is not None else kind
        nbytes = payload * (8 if dtype == torch.int64 else 4) // (len(ranks) if kind == "reduce_scatter" else 1)
        cost = (native if len(nodes) == 1 else remote).seconds("p2p" if active else kind_key,
                                                             len(active or ranks), nbytes)
        if len(nodes) > 1:
            cost *= args.remote_scale
        if counts and not active:
            cost *= max(1, max(map(sum, zip(*counts))) / max(map(sum, counts)))
        communication[phase].append((cost, nodes))
        contracts.append(dict(kind=kind_key, participants=ranks, context=context, input=payload,
                               output=inferred.return_type.shape[0], counts=counts,
                               dtype=str(dtype),
                               source="specialized synthetic routing" if counts else "static group",
                               physical_nodes=sorted(nodes)))

    for p in range(pp):
        for d in range(data):
            for e in range(ep):
                members = mesh[p, d, e, :]
                kv = 2 * local * shape.kv_heads * shape.head_dim
                contract("all_gather", members, kv, phase=f"{p}/cp_f")
                contract("reduce_scatter", members, kv * cp, phase=f"{p}/cp_b")
                if p == pp - 1:
                    router = shape.layers * local * shape.experts
                    contract("all_gather", members, router, phase=f"{p}/aux_f")
                    contract("reduce_scatter", members, router * cp, phase=f"{p}/aux_b")
            for c in range(cp):
                members = mesh[p, d, :, c]
                contract("all_gather", members, ep, dtype=torch.int64, phase=f"{p}/counts")
                routes = local * shape.top_k
                counts = []
                for source in range(ep):
                    weights = [args.routing_skew if target == 0 else 1 for target in range(ep)]
                    rows = [routes * w // sum(weights) for w in weights]
                    rows[source] += routes - sum(rows)
                    assert sum(rows) == routes
                    counts.append(tuple(n * shape.hidden for n in rows))
                counts = tuple(counts)
                contract("exchange", members, max(map(sum, counts)), counts=counts, phase=f"{p}/ep")
                transposed = tuple(zip(*counts))
                contract("exchange", members, max(map(sum, transposed)), counts=transposed, phase=f"{p}/ep")
        dense_mesh = mesh[p, :, :, :]
        contract("reduce", dense_mesh, shape.hidden * shape.expert_hidden, phase=f"{p}/reduce_dense")
        contract("broadcast", dense_mesh, shape.hidden * shape.expert_hidden, phase=f"{p}/publish_dense")
        for e in range(ep):
            expert_mesh = mesh[p, :, e, :]
            contract("reduce", expert_mesh, shape.hidden * shape.expert_hidden, phase=f"{p}/reduce_expert")
            contract("broadcast", expert_mesh, shape.hidden * shape.expert_hidden, phase=f"{p}/publish_expert")
    for p in range(pp - 1):
        for d in range(data):
            for e in range(ep):
                for c in range(cp):
                    ranks = (p * dense + (d * ep + e) * cp + c,
                             (p + 1) * dense + (d * ep + e) * cp + c)
                    payload = local * (shape.hidden + (p + 1) * (shape.layers // pp) * shape.experts)
                    counts = tuple(tuple(payload if (i, j) == (p, p + 1) else 0 for j in range(pp))
                                   for i in range(pp))
                    contract("exchange", mesh[:, d, e, c], payload, counts=counts, phase=f"{p}/pp", active=ranks)
                    contracts[-1]["active_endpoints"] = ranks
                    reversed_counts = tuple(zip(*counts))
                    contract("exchange", mesh[:, d, e, c], payload, counts=reversed_counts,
                             phase=f"{p}/pp_back", active=ranks[::-1])

    def cost(phase):
        groups = communication[phase]
        if args.contention == "independent":
            return max((seconds for seconds, _ in groups), default=0)
        busy = defaultdict(float)
        for seconds, nodes in groups:
            for node in nodes:
                busy[node] += seconds
        return max(busy.values(), default=0)

    def work(layers, sequence, local_tokens, last, skew=1):
        projections = 2 * shape.hidden * (2 * shape.q_heads + 2 * shape.kv_heads) * shape.head_dim
        experts = 6 * shape.hidden * shape.expert_hidden * shape.top_k * skew
        attention = 4 * shape.q_heads * shape.head_dim * sequence
        head = 2 * shape.hidden * shape.vocab if last else 0
        return local_tokens * (layers * (projections + experts + attention) + head)

    graph = Graph()
    order = schedule_order(pp, args.microbatches)
    for step in range(args.steps):
        for p in range(pp):
            nodes = tuple(sorted({rank // args.gpus_per_node for rank in range(p * dense, (p + 1) * dense)}))
            fabric = tuple(f"fabric/{node}" for node in nodes)
            previous = (f"{step - 1}/{p}/publish",) if step else ()
            last = p == pp - 1
            role = "last" if last else "body"
            skew = args.routing_skew * ep / (ep + args.routing_skew - 1)
            ratio = work(shape.layers // pp, tokens, local, last, skew) / work(
                baseline["shape"]["layers"] // baseline["pp"], baseline["tokens"], baseline["local_tokens"], last)
            for action, microbatch in order[p]:
                key = f"{step}/{p}/{action}/{microbatch}"
                deps = list(previous)
                if action == "F" and p:
                    deps.append(f"{step}/{p - 1}/sendF/{microbatch}")
                if action == "B":
                    deps.append(f"{step}/{p}/F/{microbatch}")
                    if not last:
                        deps.append(f"{step}/{p + 1}/sendB/{microbatch}")
                phase_cost = cost(f"{p}/cp_f") + cost(f"{p}/ep") + cost(f"{p}/counts")
                if action == "B":
                    phase_cost += cost(f"{p}/cp_b") + cost(f"{p}/ep")
                auxiliary = cost(f"{p}/aux_f" if action == "F" else f"{p}/aux_b") if last else 0
                comm = graph.add(key + "/comm", phase_cost * shape.layers / pp + auxiliary, deps, fabric,
                                 name=f"{action} CP/EP", stage=p, lane=1, step=step, microbatch=microbatch)
                phase = "forward" if action == "F" else "backward"
                graph.add(key, compute[f"{role}/{phase}"] * ratio, (comm,), (f"compute/{p}",),
                          name=phase, stage=p, lane=0, step=step, microbatch=microbatch)
                previous = (key,)
                peer = p + (1 if action == "F" else -1)
                if 0 <= peer < pp:
                    peer_nodes = {rank // args.gpus_per_node for rank in range(peer * dense, (peer + 1) * dense)}
                    graph.add(f"{step}/{p}/send{action}/{microbatch}", cost(f"{min(p, peer)}/pp"), (key,),
                              tuple(f"fabric/{node}" for node in set(nodes) | peer_nodes),
                              name=f"PP {action}", stage=p, lane=1, step=step, microbatch=microbatch, priority=-1)
            def optimizer_communication(phase):
                dense_units = shape.layers / pp * 2 * (shape.q_heads + shape.kv_heads) * shape.head_dim / shape.expert_hidden
                dense_units += shape.vocab / shape.expert_hidden * (int(p == 0) + int(last))
                return (dense_units * cost(f"{p}/{phase}_dense")
                        + shape.layers / pp * shape.experts / ep * 3 * cost(f"{p}/{phase}_expert"))

            reduced = graph.add(f"{step}/{p}/reduce", optimizer_communication("reduce"), previous, fabric,
                                 name="optimizer reduction", stage=p, lane=1, step=step)
            update = graph.add(f"{step}/{p}/update", compute[f"{role}/optimizer"] *
                                (shape.layers / pp) / (baseline["shape"]["layers"] / baseline["pp"]) * baseline["dp"] / dense,
                                (reduced,), (f"compute/{p}",), name="optimizer compute", stage=p, lane=0, step=step)
            graph.add(f"{step}/{p}/publish", optimizer_communication("publish"), (update,), fabric,
                      name="weight publication", stage=p, lane=1, step=step)
    elapsed, events = graph.run()
    result = dict(grid=dict(pp=pp, cp=cp, ep=ep, data=data, ranks=world), tokens=tokens,
                  local_tokens=local, layers=shape.layers, microbatches=args.microbatches, steps=args.steps,
                  seconds=elapsed, seconds_per_step=elapsed / args.steps, events=len(events),
                  contracts=len(contracts), numerical_evaluation=False,
                  logical_tokens_per_step=tokens * ep * data * args.microbatches,
                  contention=args.contention, remote_scale=args.remote_scale,
                  routing_skew=args.routing_skew,
                  modeled_physical_nodes=(world + args.gpus_per_node - 1) // args.gpus_per_node)
    return result, contracts, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-run", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--grid", action="append", required=True, help="PP,CP,EP,outer-data")
    parser.add_argument("--tokens-per-cp", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--microbatches", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--contention", choices=("independent", "serialized-node"), default="serialized-node")
    parser.add_argument("--remote-scale", type=float, default=1)
    parser.add_argument("--routing-skew", type=int, default=3, help="relative route weight for the hot expert destination")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens_per_cp, args.layers, args.microbatches, args.steps, args.gpus_per_node,
           args.remote_scale, args.routing_skew) <= 0:
        parser.error("dimensions, steps and scaling must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline, compute = compute_calibration(args.validated_run)
    if any(baseline["shape"][key] != value for key, value in asdict(FULL).items() if key != "layers"):
        raise ValueError("full-width compute calibration is required")
    native, remote = Calibration(args.native), Calibration(args.remote)
    args.output.mkdir(parents=True)
    results = []
    for text in args.grid:
        grid = tuple(map(int, text.split(",")))
        if len(grid) != 4 or min(grid) < 1:
            raise ValueError("grid requires four positive integers")
        result, contracts, events = stress_grid(grid, args, baseline, compute, native, remote)
        results.append(result)
        with gzip.open(args.output / f"grid-{text}.json.gz", "wt") as handle:
            json.dump(dict(result=result, contracts=contracts, traceEvents=events), handle)
    report = dict(results=results, compute_seconds=compute, shape=asdict(FULL),
                  numerical_gate=str(args.validated_run), native=str(args.native), remote=str(args.remote),
                  scope="inferred phase-level discrete-event schedule; no tensor execution beyond validated grid",
                  assumptions=["stage cohorts execute in lockstep; no within-phase compute/communication overlap",
                               "PyTorch 1F1B action order; unique microbatch labels normalized in diagnostic IR",
                               "compute scaled by analytic FLOP ratios with fixed efficiency; optimizer scaled by ownership",
                               "measured 2/4/8-rank communication interpolated in bytes; larger groups extrapolate rounds",
                               f"one hot expert destination receives {args.routing_skew} times the route weight; volume scales service time",
                               "contiguous rank placement with shared node fabric resource; no RDMA hardware measurement",
                               "specialized skewed static routing; no runtime-count HIR or request-lifetime proof",
                               "coarse optimizer reduction/publication model; no claim of predicting dmuon asynchronous overlap"])
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
