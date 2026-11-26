"""Policies: the order each rank runs its nodes in, as order edges (D5).

A policy never changes the data edges. Each one gives every rank a schedule, a total
order of its nodes that the data edges allow, and the order edges follow from it: the
compute nodes in one chain, the communication nodes in another, and each
communication node after the compute node scheduled last before it. A rank has one
compute slot and one communication slot (D7), so the two chains are what it can
overlap. ``program`` also puts each compute node after the communication node
before it: one node at a time, the order the peak of ``facts`` assumes (D11).

The members of a group must start the group's collectives in the same order, or the
simulator deadlocks; every policy here keeps the communication nodes of a rank in a
order its group shares, and ``check`` proves it before a file is written.
"""

from __future__ import annotations

import heapq
import re
from collections import defaultdict
from typing import Callable

from .facts import COLLECTIVES, Facts, Node

Edge = tuple[int, int]
Policy = Callable[[Facts], tuple[Edge, ...]]
BLOCK = re.compile(r"^mb(\d+)/stage(\d+)/([fb])/")
MICROBATCH = re.compile(r"^mb(\d+)/")


def by_rank(facts: Facts) -> dict[int, list[Node]]:
    ranks: dict[int, list[Node]] = defaultdict(list)
    for node in facts.nodes:
        ranks[node.rank].append(node)
    return ranks


def chained(schedule: list[Node], *, loose=lambda node: False, serial=False) -> list[Edge]:
    """The order edges of one rank's *schedule*.

    A *loose* communication node waits for no compute; with *serial*, a compute node
    also waits for the communication node before it, so nothing overlaps.
    """
    edges = []
    compute = comm = None
    for node in schedule:
        if node.kind == "compute":
            if compute is not None:
                edges.append((compute.id, node.id))
            if serial and comm is not None:
                edges.append((comm.id, node.id))
            compute = node
        else:
            if comm is not None:
                edges.append((comm.id, node.id))
            if compute is not None and not loose(node):
                edges.append((compute.id, node.id))
            comm = node
    return edges


def program_order(facts: Facts) -> tuple[Edge, ...]:
    """Every rank runs its nodes in program order, one at a time."""
    return tuple(edge for nodes in by_rank(facts).values() for edge in chained(nodes, serial=True))


def is_weight_gather(node: Node) -> bool:
    return node.kind == "all_gather" and "_full/" in node.name


def prefetch(facts: Facts) -> tuple[Edge, ...]:
    """Program order in two chains that overlap; a weight's all_gather waits only for the communication before it.

    So communication runs beside the compute after it, and the next layer's gather
    can run under this layer's compute.
    """
    return tuple(
        edge for nodes in by_rank(facts).values() for edge in chained(nodes, loose=is_weight_gather)
    )


def blocks(nodes: list[Node]) -> dict[tuple, list[Node]]:
    """A rank's nodes by block: ``(m, s, f|b)`` for a microbatch's pass, ``("pre",)`` and ``("post",)`` around them.

    A node of microbatch ``m`` outside a stage's pass joins the last pass of ``m`` it
    reads from (a send of the activation, the gradient's accumulation), or else the
    first pass of ``m`` that reads it (a receive, the token lookup); any other node is
    before the first pass or after the last.
    """
    key: dict[int, tuple] = {}
    for node in nodes:
        match = BLOCK.match(node.name)
        if match:
            key[node.id] = (int(match[1]), int(match[2]), match[3])
    users: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        for dep in node.deps:
            users[dep].append(node.id)

    def joined(node: Node, neighbours, at: int) -> tuple | None:
        microbatch = MICROBATCH.match(node.name)
        if not microbatch:
            return None
        found = [key[other] for other in neighbours if other in key]
        found = [each for each in found if each[0] == int(microbatch[1])]
        return found[at] if found else None

    for node in nodes:
        if node.id not in key and (found := joined(node, sorted(node.deps), -1)):
            key[node.id] = found
    for node in reversed(nodes):
        if node.id not in key and (found := joined(node, sorted(users[node.id]), 0)):
            key[node.id] = found
    out: dict[tuple, list[Node]] = defaultdict(list)
    seen_block = False
    for node in nodes:
        seen_block |= node.id in key
        out[key.get(node.id, ("post",) if seen_block else ("pre",))].append(node)
    return out


def one_f_one_b_order(keys: list[tuple], stage: int, stages: int) -> list[tuple]:
    """1F1B: ``stages - stage`` forwards first, then a backward and a forward in turn."""
    forwards = sorted(key for key in keys if key[2] == "f")
    backwards = sorted(key for key in keys if key[2] == "b")
    order, warm = [], min(stages - stage, len(forwards))
    order += forwards[:warm]
    forwards = forwards[warm:]
    while backwards:
        order.append(backwards.pop(0))
        if forwards:
            order.append(forwards.pop(0))
    return order + forwards


def one_f_one_b(facts: Facts) -> tuple[Edge, ...]:
    """Microbatch passes in 1F1B order; nodes inside a pass, and before and after the passes, in program order."""
    stages = 2
    edges = []
    for nodes in by_rank(facts).values():
        groups = blocks(nodes)
        passes = [key for key in groups if len(key) == 3]
        stage = passes[0][1] if passes else 0
        order = [("pre",)] + one_f_one_b_order(passes, stage, stages) + [("post",)]
        schedule = [node for key in order for node in groups.get(key, [])]
        edges += chained(schedule)
    return tuple(edges)


def comm_first(facts: Facts) -> tuple[Edge, ...]:
    """Program order, but a compute node that feeds communication runs before others that are ready.

    Only nodes whose data edges are met may move ahead, and only within the layer of
    the next node in program order.
    """
    feeds = {dep for node in facts.nodes if node.kind != "compute" for dep in node.deps}
    edges = []
    for nodes in by_rank(facts).values():
        edges += chained(list_schedule(nodes, feeds))
    return tuple(edges)


def layer_of(node: Node) -> str:
    return node.name.rsplit("/", 1)[0]


def list_schedule(nodes: list[Node], feeds: set[int]) -> list[Node]:
    """A topological order of *nodes* by their data edges, preferring compute nodes in *feeds*."""
    index = {node.id: node for node in nodes}
    waiting = {node.id: sum(1 for dep in node.deps if dep in index) for node in nodes}
    users: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        for dep in node.deps:
            if dep in index:
                users[dep].append(node.id)
    program = [node.id for node in nodes]
    done: set[int] = set()
    ready_compute: list[tuple[int, int]] = []
    ready_comm: list[int] = []
    for node in nodes:
        if waiting[node.id] == 0:
            push(node, ready_compute, ready_comm, feeds)
    out: list[Node] = []
    cursor = 0
    while len(out) < len(nodes):
        while program[cursor] in done:
            cursor += 1
        head = index[program[cursor]]
        if head.kind != "compute":
            # Communication keeps program order: the group shares it.
            if waiting[head.id] != 0 or head.id not in ready_comm:
                raise ValueError(f"comm_first: {head.name} is not ready in program order")
            ready_comm.remove(head.id)
            chosen = head
        else:
            # The head is ready: every node before it in program order is done.
            layer = layer_of(head)
            feeding = [
                item for item in ready_compute if item[0] == 0 and layer_of(index[item[1]]) == layer
            ]
            pick = (
                min(feeding)
                if feeding
                else next(item for item in ready_compute if item[1] == head.id)
            )
            ready_compute.remove(pick)
            heapq.heapify(ready_compute)
            chosen = index[pick[1]]
        out.append(chosen)
        done.add(chosen.id)
        for user in users[chosen.id]:
            waiting[user] -= 1
            if waiting[user] == 0:
                push(index[user], ready_compute, ready_comm, feeds)
    return out


def push(node: Node, ready_compute: list, ready_comm: list, feeds: set[int]) -> None:
    if node.kind == "compute":
        heapq.heappush(ready_compute, (0 if node.id in feeds else 1, node.id))
    else:
        ready_comm.append(node.id)


POLICIES: dict[str, Policy] = {
    "program": program_order,
    "prefetch": prefetch,
    "1f1b": one_f_one_b,
    "comm_first": comm_first,
}


def check(facts: Facts, edges: tuple[Edge, ...], name: str) -> None:
    """No cycle through data and order edges; every group's members start its collectives in one order."""
    following: dict[int, list[int]] = defaultdict(list)
    degree = {node.id: 0 for node in facts.nodes}
    for node in facts.nodes:
        for dep in node.deps:
            following[dep].append(node.id)
            degree[node.id] += 1
    for src, dst in edges:
        following[src].append(dst)
        degree[dst] += 1
    stack = [nid for nid, count in degree.items() if count == 0]
    seen = 0
    while stack:
        nid = stack.pop()
        seen += 1
        for nxt in following[nid]:
            degree[nxt] -= 1
            if degree[nxt] == 0:
                stack.append(nxt)
    if seen != len(facts.nodes):
        stuck = [facts.nodes[nid].name for nid, count in degree.items() if count][:6]
        raise ValueError(f"policy {name!r} makes a cycle: {' -> '.join(stuck)}")
    order = rank_orders(facts, edges)
    by_group: dict[tuple[int, ...], dict[int, list[Node]]] = defaultdict(lambda: defaultdict(list))
    for rank, nodes in order.items():
        for node in nodes:
            if node.kind in COLLECTIVES:
                by_group[node.group][rank].append(node)
    for group, ranks in by_group.items():
        if len(group) < 2:
            raise ValueError(f"policy {name!r}: group {group} has one member")
        first, *others = sorted(ranks)
        for other in others:
            for position, (a, b) in enumerate(zip(ranks[first], ranks[other])):
                if (a.kind, a.bytes) != (b.kind, b.bytes) or strip_rank(a.name) != strip_rank(
                    b.name
                ):
                    raise ValueError(
                        f"policy {name!r}: ranks {first} and {other} order the collectives of group {group} "
                        f"differently at position {position}: {a.name}, {b.name}"
                    )


def strip_rank(name: str) -> str:
    return name.split("#", 1)[0]


def rank_orders(facts: Facts, edges: tuple[Edge, ...]) -> dict[int, list[Node]]:
    """Each rank's communication nodes in the order its communication chain gives."""
    chain = {
        src: dst
        for src, dst in edges
        if facts.nodes[src].kind != "compute" and facts.nodes[dst].kind != "compute"
    }
    has_prev = set(chain.values())
    out: dict[int, list[Node]] = defaultdict(list)
    for node in facts.nodes:
        if node.kind != "compute" and node.id not in has_prev:
            nid = node.id
            while nid is not None:
                out[facts.nodes[nid].rank].append(facts.nodes[nid])
                nid = chain.get(nid)
    return out


__all__ = ["POLICIES", "Policy", "check"]
