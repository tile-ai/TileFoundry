"""Write ``graph.json`` for every configuration, recompute choice and policy.

    python -m examples.qwen3_training.export --out runs [--config cp ...] [--recompute none ...] [--policy program ...]

A choice left out means all of its values: 5 configurations x 3 recompute choices x 4
policies = 60 directories ``runs/<config>/<recompute>/<policy>/``. The structure of a
(configuration, recompute) pair is read once (``facts.py``) and written once per policy;
only the order edges differ between the four files.

Times. A FlashAttention node takes FA4's measured time (``calibration/compute.json``):
FA4 is the whole kernel, and the HIR's roofline also counts the masked scores. A kernel
measured by its GEMMs takes the GEMMs' time, never less than its fused roofline; under
``layer`` its backward runs its forward again, so it also takes the forward's GEMMs.
Any other compute node takes its fused roofline (D62, D69). A communication node takes the
time ``calibration/assumed.json`` states for its kind, group size and bytes; the
simulator cannot be told a time, so the node carries ``sim_bytes``, the size the
simulator's network model spends that time on (D6, ``calibration/simulator.json``).
The communication times are assumed, not measured.
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import multiprocessing
import sys
import time
from pathlib import Path

from tilefoundry import target

from . import step as steps
from .configs import AXES, CONFIGS, FULL, GPUS
from .facts import COLLECTIVES, Facts, coords, facts
from .model import TARGET
from .policies import POLICIES, Policy, check

HERE = Path(__file__).parent
CALIBRATION = HERE / "calibration"
RECORD_KIND = {
    "all_gather": "all_gather",
    "reduce_scatter": "reduce_scatter",
    "all_reduce": "all_reduce",
    "all_to_all": "exchange",
    "send": "p2p",
    "recv": "p2p",
}


class Calibration:
    """Communication times: what a node should take, and the size that makes the simulator take it."""

    def __init__(
        self,
        assumed: Path = CALIBRATION / "assumed.json",
        simulator: Path = CALIBRATION / "simulator.json",
    ):
        table = json.loads(assumed.read_text())
        self.transport = table["transport"]
        self.points: dict[tuple[str, int], list[tuple[int, float]]] = {}
        for record in table["records"]:
            self.points.setdefault((record["kind"], record["size"]), []).append(
                (record["bytes"], record["seconds"])
            )
        for points in self.points.values():
            points.sort()
        self.response = {
            (item["kind"], item["size"]): (item["ns_at_zero"], item["ns_per_byte"])
            for item in json.loads(simulator.read_text())["fits"]
        }

    def seconds(self, kind: str, group_size: int, nbytes: int) -> float:
        """The assumed time of one *kind* on *group_size* ranks moving *nbytes*, interpolated in bytes."""
        points = self.points.get((RECORD_KIND[kind], group_size))
        if not points or len(points) < 2:
            raise ValueError(
                f"calibration {self.transport!r} has no record for {kind} with {group_size} ranks"
            )
        sizes = [size for size, _ in points]
        at = min(max(bisect.bisect_left(sizes, nbytes), 1), len(points) - 1)
        (x0, y0), (x1, y1) = points[at - 1], points[at]
        return y0 + (y1 - y0) * (nbytes - x0) / (x1 - x0)

    def bytes_for(self, kind: str, group_size: int, nbytes: int) -> int:
        """The size to give the simulator so that its network model spends the assumed time."""
        key = ("p2p" if kind in ("send", "recv") else kind, group_size)
        if key not in self.response:
            raise ValueError(f"simulator.json has no fit for {kind} with {group_size} ranks")
        base, slope = self.response[key]
        wanted = self.seconds(kind, group_size, nbytes) * 1e9
        return max(1, round((wanted - base) / slope))


FORWARD_AGAIN = {
    "none": (),
    "regather": (),
    "layer": ("AttnIn", "AttnOut", "MoE", "Router", "Experts", "Combine"),
}
"""The GEMM kernels whose backward computes their forward again, by recompute choice.

``Layer<s>`` names the layer's whole subtree, so each kernel in it recomputes its inside
from its saved inputs. FlashAttention and the head are recomputed under every choice;
their measured backward already includes it.
"""


class ComputeTimes:
    """Compute times: the measured kernels of ``compute.json``, the fused roofline for the rest."""

    def __init__(self, path: Path = CALIBRATION / "compute.json", tokens: int = FULL.tokens):
        table = json.loads(path.read_text())
        # The kernels were measured at the full size; another size has no measurement.
        self.tokens = tokens
        measured = tokens == FULL.tokens
        self.source = (
            f"{table['source']} on {table['device']}, {table['dtype']}"
            if measured
            else f"roofline only: measured kernels are for {FULL.tokens} tokens"
        )
        self.entries = table["entries"] if measured else {}
        throughput = TARGET.get_facts(target.ThroughputFacts)
        peaks = {dtype.name: flops for dtype, flops in throughput.peak_flops_per_second}
        self.flops_per_second = peaks["bf16"] / GPUS
        self.bytes_per_second = throughput.memory_bandwidth_bytes_per_second / GPUS

    def roofline(self, flops: int, nbytes: int) -> int:
        return int(max(flops / self.flops_per_second, nbytes / self.bytes_per_second) * 1e9)

    def time(self, config: str, recompute: str, node) -> tuple[int, str]:
        roofline = self.roofline(node.flops, node.bytes)
        entries = self.entries.get(config, {})
        key = f"{node.op}/{node.phase}"
        if node.op == "FlashAttention":
            key += f"/cp{coords(node.rank, CONFIGS[config])[AXES.index('cp')]}"
        measured = entries.get(key)
        if measured is None:
            return max(roofline, 1), "roofline"
        if node.op == "FlashAttention":
            return measured, "measured"
        if node.phase == "b" and node.op in FORWARD_AGAIN[recompute]:
            measured += entries.get(f"{node.op}/f", 0)
        return (measured, "measured") if measured >= roofline else (roofline, "roofline")


def export(
    found: Facts, policy: str, calibration: Calibration, compute: ComputeTimes, out_dir: Path
) -> Path:
    """Write ``graph.json`` for *found* under *policy* into *out_dir*."""
    order = POLICIES[policy](found)
    check(found, order, policy)
    following: dict[int, list[int]] = {}
    for src, dst in order:
        following.setdefault(dst, []).append(src)
    groups: dict[tuple[int, ...], str] = {}
    nodes = []
    for node in found.nodes:
        item = {
            "id": node.id,
            "rank": node.rank,
            "name": node.name,
            "kind": node.kind,
            "op": node.op,
            "phase": node.phase,
            "flops": node.flops,
            "bytes": node.bytes,
        }
        if node.kind == "compute":
            item["ns"], item["ns_source"] = compute.time(found.config, found.recompute, node)
            item["sim_bytes"] = 0
        else:
            size = len(node.group) if node.kind in COLLECTIVES else 2
            item["ns"] = int(calibration.seconds(node.kind, size, node.bytes) * 1e9)
            item["ns_source"] = calibration.transport
            item["sim_bytes"] = calibration.bytes_for(node.kind, size, node.bytes)
        if node.kind in COLLECTIVES:
            item["group"] = groups.setdefault(node.group, f"g{len(groups)}")
        elif node.kind in ("send", "recv"):
            item["src"], item["dst"], item["tag"] = node.group[0], node.group[1], node.tag
        item["data_deps"] = list(node.deps)
        item["order_deps"] = sorted(following.get(node.id, ()))
        nodes.append(item)
    graph = {
        "config": found.config,
        "recompute": found.recompute,
        "policy": policy,
        "tokens": compute.tokens,
        "ranks": found.ranks,
        "mesh": {axis: CONFIGS[found.config].size_of(axis) for axis in AXES},
        "peak": found.peak,
        "peak_rank": found.peak_rank,
        "capacity": found.capacity,
        "communication_bytes": found.communication_bytes,
        "calibration": {"communication": calibration.transport, "compute": compute.source},
        "groups": {name: list(group) for group, name in groups.items()},
        "nodes": nodes,
        "values": [
            {
                "id": value.id,
                "rank": value.rank,
                "name": value.name,
                "bytes": value.bytes,
                "producer": value.producer,
                "consumers": list(value.consumers),
                "persistent": value.persistent,
            }
            for value in found.values
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "graph.json"
    path.write_text(json.dumps(graph, separators=(",", ":")))
    return path


def one(job) -> list[str]:
    """Read one (configuration, recompute) pair and write its policies; the lines to print."""
    config, recompute, policies, out, tokens = job
    started = time.time()
    try:
        found = facts(
            steps.build_step(CONFIGS[config], dataclasses.replace(FULL, tokens=tokens), recompute),
            CONFIGS[config],
            recompute,
        )
        calibration, compute = Calibration(), ComputeTimes(tokens=tokens)
        lines = []
        for policy in policies:
            path = export(found, policy, calibration, compute, out / config / recompute / policy)
            lines.append(f"{path}  nodes {len(found.nodes)}  {path.stat().st_size} bytes")
        lines.append(
            f"{config}/{recompute}: facts {found.seconds} s, total {time.time() - started:.1f} s"
        )
        return lines
    except Exception as error:  # one pair failing must not stop the others
        return [f"FAILED {config}/{recompute}: {type(error).__name__}: {error}"]


def peak_at(config: str, recompute: str, tokens: int) -> int:
    """``Facts.peak`` of the step with *tokens* tokens per sequence."""
    size = dataclasses.replace(FULL, tokens=tokens)
    return facts(
        steps.build_step(CONFIGS[config], size, recompute), CONFIGS[config], recompute
    ).peak


def fit(job, analyses: int = 8) -> dict:
    """The largest ``tokens`` whose step fits in a GPU's gmem, by analysis only (D42).

    ``tokens`` moves in steps of ``16 * cp``, the smallest change that keeps a whole
    expert capacity on every cp part. The peak is close to affine in ``tokens``, so
    each guess is the secant through the last two analyses; the search ends at a
    fitting ``tokens`` whose next step does not fit, or after *analyses* analyses.
    """
    config, recompute = job
    step = 16 * CONFIGS[config].cp
    capacity = TARGET.get_facts(target.MemoryHierarchyFacts).explicit("gmem").capacity_bytes
    seen: dict[int, int] = {}

    def analysed(tokens: int) -> int:
        if tokens not in seen:
            seen[tokens] = peak_at(config, recompute, tokens)
        return seen[tokens]

    low, high = FULL.tokens, 2 * FULL.tokens
    while len(seen) < analyses:
        a, b = analysed(low), analysed(high)
        fitting = [tokens for tokens, peak in seen.items() if peak <= capacity]
        too_big = [tokens for tokens, peak in seen.items() if peak > capacity]
        if fitting and too_big and min(too_big) - max(fitting) == step:
            break
        guess = low + (capacity - a) * (high - low) / (b - a) if b != a else high + step
        guess = max(step, int(guess) // step * step)
        if fitting and guess <= max(fitting):
            guess = max(fitting) + step
        if too_big and guess >= min(too_big):
            guess = min(too_big) - step
        low, high = high, guess
        if low == high:
            high = low + step
    fitting = [tokens for tokens, peak in seen.items() if peak <= capacity]
    too_big = [tokens for tokens, peak in seen.items() if peak > capacity]
    best = max(fitting, default=None)
    return {
        "config": config,
        "recompute": recompute,
        "tokens": best,
        "peak": seen.get(best),
        "next": min(too_big, default=None),
        "next_peak": seen.get(min(too_big, default=-1)),
        "step": step,
        "capacity": capacity,
        "analyses": len(seen),
        "found": bool(fitting and too_big and min(too_big) - max(fitting) == step),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", nargs="+", choices=tuple(CONFIGS), default=tuple(CONFIGS))
    parser.add_argument("--recompute", nargs="+", choices=steps.RECOMPUTE, default=steps.RECOMPUTE)
    parser.add_argument("--policy", nargs="+", choices=tuple(POLICIES), default=tuple(POLICIES))
    parser.add_argument("--jobs", type=int, default=5, help="pairs read at once")
    parser.add_argument(
        "--tokens",
        type=int,
        default=FULL.tokens,
        help="tokens per sequence (default: the full size)",
    )
    parser.add_argument(
        "--fit", action="store_true", help="write <out>/fit.json: the largest tokens that fit (D42)"
    )
    args = parser.parse_args(argv)
    if args.fit:
        pairs = [(config, recompute) for config in args.config for recompute in args.recompute]
        started = time.time()
        with multiprocessing.get_context("spawn").Pool(min(args.jobs, len(pairs))) as pool:
            found = pool.map(fit, pairs)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "fit.json").write_text(json.dumps(found, indent=1) + "\n")
        for row in found:
            print(json.dumps(row), flush=True)
        print(f"fit: {len(pairs)} pairs in {time.time() - started:.1f} s", flush=True)
        return 0 if all(row["found"] for row in found) else 1
    jobs = [
        (config, recompute, args.policy, args.out, args.tokens)
        for config in args.config
        for recompute in args.recompute
    ]
    started = time.time()
    failed = False
    with multiprocessing.get_context("spawn").Pool(min(args.jobs, len(jobs))) as pool:
        for lines in pool.imap_unordered(one, jobs):
            for line in lines:
                failed |= line.startswith("FAILED")
                print(line, flush=True)
    print(
        f"export: {len(jobs) * len(args.policy)} graphs in {time.time() - started:.1f} s",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Calibration", "ComputeTimes", "Policy", "export", "main"]
