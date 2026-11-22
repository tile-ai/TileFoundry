"""Write one run's ``graph.json`` as Chakra execution traces for ASTRA-sim.

    python examples/qwen3_training/sim/to_chakra.py runs/<config>/<recompute>/<policy>

Runs in the simulator's venv: it imports ``chakra``, never ``tilefoundry`` (D31).
It writes ``trace/trace.<rank>.et`` and ``trace/comm_group.json`` next to
``graph.json``:

======================================  ==========================================================
graph.json                              Chakra
======================================  ==========================================================
compute                                 COMP_NODE, ``duration_micros = ns // 1000``, at least 1
all_gather, reduce_scatter, all_reduce,  COMM_COLL_NODE: ``comm_type``, ``comm_size = sim_bytes``,
all_to_all                              ``pg_name``
send, recv                              COMM_SEND_NODE, COMM_RECV_NODE: ``comm_src``, ``comm_dst``,
                                        ``comm_tag``, ``comm_size = sim_bytes``
data_deps / order_deps                  data_deps / ctrl_deps
name                                    ``<node type>_<name>``: chakra_timeline_visualizer reads the
                                        type from the name (D8)
======================================  ==========================================================

Attribute names, types and units follow ASTRA-sim at the pinned commit
(``astra-sim/workload/Workload.cc``, Chakra ``feeder_v3/et_feeder_node_attr.h``).
A rank's nodes are written in an order their edges allow, lowest id first, so no
edge points forward in a file.
"""

from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

from chakra.schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    REDUCE_SCATTER,
    GlobalMetadata,
    Node,
)
from chakra.schema.protobuf.et_def_pb2 import AttributeProto as Attr
from chakra.src.third_party.utils.protolib import encodeMessage

COLLECTIVE = {
    "all_gather": ALL_GATHER,
    "reduce_scatter": REDUCE_SCATTER,
    "all_reduce": ALL_REDUCE,
    "all_to_all": ALL_TO_ALL,
}
TYPE_NAME = {
    COMP_NODE: "COMP_NODE",
    COMM_COLL_NODE: "COMM_COLL_NODE",
    COMM_SEND_NODE: "COMM_SEND_NODE",
    COMM_RECV_NODE: "COMM_RECV_NODE",
}


class Builder:
    """One node list per rank and one table of communicator groups."""

    def __init__(self, ranks: int):
        self.ranks = ranks
        self.nodes: list[list[Node]] = [[] for _ in range(ranks)]
        self.groups: dict[tuple[int, ...], int] = {}

    def group(self, ranks) -> str:
        """The ``pg_name`` of the group of *ranks*: its id in ``comm_group.json``."""
        key = tuple(sorted(ranks))
        if key not in self.groups:
            self.groups[key] = len(self.groups) + 1
        return str(self.groups[key])

    def add(
        self, rank: int, name: str, type_: int, deps=(), ctrl=(), micros: int = 0, attrs=()
    ) -> int:
        """Add a node to *rank*'s list; its id is its position there."""
        if "," in name or "=" in name:
            raise ValueError(
                f"node name {name!r} has a ',' or '=': the timeline tool splits on them"
            )
        node = Node()
        node.id = len(self.nodes[rank])
        node.name = f"{TYPE_NAME[type_]}_{name}"
        node.type = type_
        node.duration_micros = int(micros)
        node.data_deps.extend(deps)
        node.ctrl_deps.extend(ctrl)
        node.attr.append(Attr(name="is_cpu_op", bool_val=False))
        node.attr.extend(attrs)
        self.nodes[rank].append(node)
        return node.id

    def compute(self, rank: int, name: str, micros: int, deps=(), ctrl=()) -> int:
        return self.add(rank, name, COMP_NODE, deps, ctrl, micros)

    def collective(
        self, rank: int, name: str, kind: str, size: int, ranks, deps=(), ctrl=()
    ) -> int:
        attrs = [
            Attr(name="comm_type", int64_val=COLLECTIVE[kind]),
            Attr(name="comm_size", int64_val=int(size)),
            Attr(name="pg_name", string_val=self.group(ranks)),
        ]
        return self.add(rank, name, COMM_COLL_NODE, deps, ctrl, 0, attrs)

    def point(
        self, kind: str, src: int, dst: int, name: str, size: int, tag: int, deps=(), ctrl=()
    ) -> int:
        """A send on *src* or a recv on *dst*, by *kind*."""
        attrs = [
            Attr(name="comm_src", int64_val=src),
            Attr(name="comm_dst", int64_val=dst),
            Attr(name="comm_tag", int64_val=tag),
            Attr(name="comm_size", int64_val=int(size)),
        ]
        type_, rank = (COMM_SEND_NODE, src) if kind == "send" else (COMM_RECV_NODE, dst)
        return self.add(rank, name, type_, deps, ctrl, 0, attrs)

    def write(self, out_dir: Path, name: str = "trace") -> Path:
        """Write ``<name>.<rank>.et`` for each rank and ``comm_group.json``; the workload prefix."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for rank, nodes in enumerate(self.nodes):
            with open(out_dir / f"{name}.{rank}.et", "wb") as file:
                encodeMessage(file, GlobalMetadata(version="0.0.4"))
                for node in nodes:
                    encodeMessage(file, node)
        groups = {str(gid): list(ranks) for ranks, gid in self.groups.items()}
        (out_dir / "comm_group.json").write_text(json.dumps(groups, indent=1) + "\n")
        return out_dir / name


def file_order(nodes: list[dict]) -> list[dict]:
    """*nodes* of one rank in an order their data and order edges allow, lowest id first."""
    index = {node["id"]: node for node in nodes}
    waiting = {node["id"]: 0 for node in nodes}
    users: dict[int, list[int]] = {node["id"]: [] for node in nodes}
    for node in nodes:
        for dep in (*node["data_deps"], *node["order_deps"]):
            if dep not in index:
                raise ValueError(f"node {node['name']!r} depends on node {dep} of another rank")
            waiting[node["id"]] += 1
            users[dep].append(node["id"])
    ready = [nid for nid, count in waiting.items() if count == 0]
    heapq.heapify(ready)
    out = []
    while ready:
        nid = heapq.heappop(ready)
        out.append(index[nid])
        for user in users[nid]:
            waiting[user] -= 1
            if waiting[user] == 0:
                heapq.heappush(ready, user)
    if len(out) != len(nodes):
        raise ValueError("the edges of a rank make a cycle")
    return out


def convert(graph: dict) -> Builder:
    """The Chakra nodes of *graph*, one list per rank."""
    builder = Builder(graph["ranks"])
    groups = graph["groups"]
    by_rank: list[list[dict]] = [[] for _ in range(graph["ranks"])]
    for node in graph["nodes"]:
        by_rank[node["rank"]].append(node)
    for rank, nodes in enumerate(by_rank):
        local: dict[int, int] = {}
        for node in file_order(nodes):
            deps = [local[dep] for dep in node["data_deps"]]
            ctrl = [local[dep] for dep in node["order_deps"]]
            kind = node["kind"]
            if kind == "compute":
                nid = builder.compute(rank, node["name"], max(1, node["ns"] // 1000), deps, ctrl)
            elif kind in COLLECTIVE:
                ranks = groups[node["group"]]
                nid = builder.collective(
                    rank, node["name"], kind, node["sim_bytes"], ranks, deps, ctrl
                )
            else:
                nid = builder.point(
                    kind,
                    node["src"],
                    node["dst"],
                    node["name"],
                    node["sim_bytes"],
                    node["tag"],
                    deps,
                    ctrl,
                )
            local[node["id"]] = nid
    return builder


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: to_chakra.py <run directory with graph.json>", file=sys.stderr)
        return 2
    run = Path(args[0])
    graph = json.loads((run / "graph.json").read_text())
    workload = convert(graph).write(run / "trace")
    print(workload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
