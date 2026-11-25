"""Join each simulated run with its graph: one timeline, and one index of all runs.

    python -m examples.qwen3_training.report runs

For each ``runs/<config>/<recompute>/<policy>/`` with a ``graph.json``, this reads
the simulator's ``sim.chrome.json`` (D47) and writes ``timeline.chrome.json``: for
each rank the simulator's compute and communication lanes, unchanged, and a memory
lane computed from the graph's values under the simulated times (D11, D44). Then it
writes ``runs/report.md``, one row per run.

Memory under a timeline: a value takes its bytes from the start of the node that
produces it to the end of the last node that reads it; a parameter (producer -1)
from the start of the step, a persistent value to its end. The granularity is a
node: a kernel's own intermediate values are not seen (D62).

Standard library only: this file does not import ``tilefoundry``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

LANES = {3: "compute", 4: "communication"}
MEMORY_LANE = 5
TYPE_PREFIXES = ("COMP_NODE_", "COMM_COLL_NODE_", "COMM_SEND_NODE_", "COMM_RECV_NODE_")


@dataclass(frozen=True)
class Event:
    rank: int
    name: str
    start: int  # ns
    end: int  # ns
    lane: int


@dataclass
class Peak:
    rank: int
    bytes: int = 0
    at: int = 0  # ns
    node: str = ""
    largest: list[tuple[str, int]] = field(default_factory=list)


def strip_type(name: str) -> str:
    for prefix in TYPE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    raise ValueError(f"sim.chrome.json names node {name!r} without a Chakra node type")


def timeline(run: Path) -> list[Event]:
    """The simulated nodes of *run*: rank, name, start and end in ns (``sim.chrome.json``)."""
    trace = json.loads((run / "sim.chrome.json").read_text())
    events = []
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        start = round(event["ts"] * 1000)
        events.append(
            Event(
                event["pid"],
                strip_type(event["name"]),
                start,
                start + round(event["dur"] * 1000),
                event["tid"],
            )
        )
    return events


def matched(events: list[Event], graph: dict) -> dict[tuple[int, str], Event]:
    """Each event by ``(rank, name)``; every name must be a node of *graph*."""
    names = {(node["rank"], node["name"]) for node in graph["nodes"]}
    by_key = {}
    for event in events:
        key = (event.rank, event.name)
        if key not in names:
            raise ValueError(
                f"sim.chrome.json names node '{event.name}' that graph.json does not have"
            )
        by_key[key] = event
    return by_key


def memory(
    events: dict[tuple[int, str], Event], graph: dict
) -> dict[int, list[tuple[int, int, str, str]]]:
    """Each rank's memory changes: ``(time, +/- bytes, value name, node name)``, in time order.

    At one time, a free comes before an allocation: the node that ends no longer
    holds what the node that starts allocates.
    """
    node_event = {node["id"]: events[(node["rank"], node["name"])] for node in graph["nodes"]}
    step_end = max((event.end for event in events.values()), default=0)
    changes: dict[int, list[tuple[int, int, str, str]]] = {}
    for value in graph["values"]:
        if value["producer"] < 0:
            start, node = 0, "step start"
        else:
            producer = node_event[value["producer"]]
            start, node = producer.start, producer.name
        if value["persistent"]:
            end = step_end
        elif value["consumers"]:
            end = max(node_event[consumer].end for consumer in value["consumers"])
        else:
            end = node_event[value["producer"]].end if value["producer"] >= 0 else start
        rank = changes.setdefault(value["rank"], [])
        rank.append((start, value["bytes"], value["name"], node))
        rank.append((end, -value["bytes"], value["name"], node))
    for rank in changes.values():
        rank.sort(key=lambda change: (change[0], change[1] > 0))
    return changes


def peak_under(changes: dict[int, list[tuple[int, int, str, str]]]) -> dict[int, Peak]:
    """Each rank's peak under the timeline, where it happens and its 3 largest values then."""
    peaks = {}
    for rank, steps in changes.items():
        peak, used, live = Peak(rank), 0, {}
        for time, delta, name, node in steps:
            used += delta
            if delta > 0:
                live[name] = live.get(name, 0) + delta
            else:
                live[name] -= -delta
                if not live[name]:
                    del live[name]
            if used > peak.bytes:
                peak.bytes, peak.at, peak.node = used, time, node
                peak.largest = sorted(live.items(), key=lambda item: -item[1])[:3]
        peaks[rank] = peak
    return peaks


def merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def overlap(events: list[Event], rank: int) -> int:
    """The time *rank* computes and communicates at once, in ns."""
    compute = merged([(e.start, e.end) for e in events if e.rank == rank and e.lane == 3])
    comm = merged([(e.start, e.end) for e in events if e.rank == rank and e.lane == 4])
    total, i, j = 0, 0, 0
    while i < len(compute) and j < len(comm):
        total += max(0, min(compute[i][1], comm[j][1]) - max(compute[i][0], comm[j][0]))
        if compute[i][1] < comm[j][1]:
            i += 1
        else:
            j += 1
    return total


def label(rank: int, mesh: dict[str, int] | None) -> str:
    if not mesh:
        return f"rank {rank}"
    coords, rest = [], rank
    for axis, size in reversed(list(mesh.items())):
        rest, index = divmod(rest, size)
        coords.append(f"{axis}={index}")
    return f"rank {rank} ({', '.join(reversed(coords))})"


def write_timeline(run: Path, graph: dict, changes, peaks: dict[int, Peak]) -> Path:
    """``timeline.chrome.json``: the simulator's events, lane names, a memory counter and a peak mark per rank."""
    trace = json.loads((run / "sim.chrome.json").read_text())
    events = trace["traceEvents"]
    nodes = {(node["rank"], node["name"]): node for node in graph["nodes"]}
    for event in events:
        if event.get("ph") == "X":
            event["name"] = strip_type(event["name"])
            node = nodes[(event["pid"], event["name"])]
            event["cat"] = node["kind"]
            event["args"].update(kind=node["kind"], bytes=node["bytes"])
    capacity = graph["capacity"]
    for rank in range(graph["ranks"]):
        events.append(
            {
                "ph": "M",
                "pid": rank,
                "name": "process_name",
                "args": {"name": label(rank, graph.get("mesh"))},
            }
        )
        events.append(
            {"ph": "M", "pid": rank, "name": "process_sort_index", "args": {"sort_index": rank}}
        )
        for tid, name in (*LANES.items(), (MEMORY_LANE, "memory")):
            events.append(
                {"ph": "M", "pid": rank, "tid": tid, "name": "thread_name", "args": {"name": name}}
            )
        used = 0
        for time, delta, _, _ in changes.get(rank, ()):
            used += delta
            events.append(
                {
                    "ph": "C",
                    "pid": rank,
                    "tid": MEMORY_LANE,
                    "name": "memory",
                    "ts": time / 1000,
                    "args": {"used": used, "capacity": capacity},
                }
            )
        peak = peaks.get(rank)
        if peak is not None:
            events.append(
                {
                    "ph": "i",
                    "s": "p",
                    "pid": rank,
                    "tid": MEMORY_LANE,
                    "ts": peak.at / 1000,
                    "name": "OOM" if peak.bytes > capacity else "peak",
                    "args": {"bytes": peak.bytes, "node": peak.node, "largest": dict(peak.largest)},
                }
            )
    trace["otherData"] = {
        "config": graph["config"],
        "recompute": graph["recompute"],
        "policy": graph["policy"],
        "tokens": graph.get("tokens"),
        "communication": f"{graph['calibration']['communication']} times (D6); one communication slot per rank (D7)",
        "compute": graph["calibration"]["compute"],
        "memory": "per node: a kernel's own intermediate values are not counted",
        "ts": "microseconds",
    }
    path = run / "timeline.chrome.json"
    path.write_text(json.dumps(trace, separators=(",", ":")))
    return path


@dataclass
class Row:
    config: str
    recompute: str
    policy: str
    capacity: int
    peak: int | None = None
    step: int | None = None
    overlap: int | None = None
    communication: int | None = None
    timeline: str = ""
    problem: str = ""


def report_run(run: Path, root: Path) -> Row:
    graph = json.loads((run / "graph.json").read_text())
    row = Row(graph["config"], graph["recompute"], graph["policy"], graph["capacity"])
    row.communication = graph["communication_bytes"]
    if not (run / "sim.chrome.json").exists():
        row.problem = "not run"
        return row
    events = timeline(run)
    by_key = matched(events, graph)
    missing = [node for node in graph["nodes"] if (node["rank"], node["name"]) not in by_key]
    if missing:
        stuck = sorted({node["rank"] for node in missing})
        last = {
            rank: max((e for e in events if e.rank == rank), key=lambda e: e.end, default=None)
            for rank in stuck
        }
        row.problem = "deadlock: " + "; ".join(
            f"rank {rank} after {event.name if event else 'nothing'}"
            for rank, event in last.items()
        )
        return row
    changes = memory(by_key, graph)
    peaks = peak_under(changes)
    top = max(peaks.values(), key=lambda peak: peak.bytes)
    row.peak = top.bytes
    row.step = max(event.end for event in events)
    last_rank = max(events, key=lambda event: event.end).rank
    row.overlap = overlap(events, last_rank)
    row.timeline = str(write_timeline(run, graph, changes, peaks).relative_to(root))
    if graph["policy"] == "program" and top.bytes != graph["peak"]:
        row.problem = f"peak mismatch: {top.bytes - graph['peak']:+d} bytes"
    return row


def gigabytes(value: int | None) -> str:
    return "" if value is None else f"{value / 1e9:.2f} GB"


def milliseconds(value: int | None) -> str:
    return "" if value is None else f"{value / 1e6:.2f} ms"


def fit_table(path: Path) -> list[str]:
    """The largest tokens that fit, from ``export.py --fit`` (D42); nothing when it was not run."""
    if not path.exists():
        return []
    lines = [
        "",
        "## The largest tokens that fit (analysis only, D42)",
        "",
        "| config | recompute | tokens | peak | next tokens | peak there | analyses |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(
        json.loads(path.read_text()), key=lambda row: (row["config"], row["recompute"])
    ):
        lines.append(
            f"| {row['config']} | {row['recompute']} | {row['tokens']} | {gigabytes(row['peak'])} "
            f"| {row['next']} | {gigabytes(row['next_peak'])} | {row['analyses']} |"
        )
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path)
    args = parser.parse_args(argv)
    rows = [report_run(graph.parent, args.runs) for graph in sorted(args.runs.rglob("graph.json"))]
    rows.sort(key=lambda row: (row.config, row.recompute, row.policy))
    lines = [
        "# Simulated training steps",
        "",
        "Communication times are assumed (`calibration/assumed.json`, D6); each rank has one",
        "communication slot (D7); memory is per node, without a kernel's own intermediate values.",
        "Overlap is the time the last rank to finish computes and communicates at once.",
        "",
        "| config | recompute | policy | peak (largest rank) | capacity | fits | step | overlap | "
        "communication (all ranks) | timeline |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        fits = "" if row.peak is None else ("yes" if row.peak <= row.capacity else "OOM")
        step = row.problem if row.problem and row.step is None else milliseconds(row.step)
        if row.problem and row.step is not None:
            step += f" ({row.problem})"
        lines.append(
            f"| {row.config} | {row.recompute} | {row.policy} | {gigabytes(row.peak)} | {gigabytes(row.capacity)} "
            f"| {fits} | {step} | {milliseconds(row.overlap)} | {gigabytes(row.communication)} | {row.timeline} |"
        )
    lines += fit_table(args.runs / "fit.json")
    (args.runs / "report.md").write_text("\n".join(lines) + "\n")
    print(args.runs / "report.md")
    problems = [row for row in rows if row.problem]
    for row in problems:
        print(f"{row.config}/{row.recompute}/{row.policy}: {row.problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
