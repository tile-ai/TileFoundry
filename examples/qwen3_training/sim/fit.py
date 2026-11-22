"""Fit the simulator's time for one communication alone, so a node can ask it for a time.

    SIM_ROOT=<install dir> $SIM_ROOT/venv/bin/python examples/qwen3_training/sim/fit.py

ASTRA-sim takes a size, not a time (D6). For each kind and group size, this runs
one communication on every group of the 8-rank network at three sizes, fits
``ns = ns_at_zero + ns_per_byte * size`` through the smallest and the largest,
checks the middle one against the line, and writes ``calibration/simulator.json``.
``export.py`` inverts the line: the size that makes the simulator spend the time
``calibration/assumed.json`` states.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from to_chakra import Builder

HERE = Path(__file__).parent
OUT = HERE.parent / "calibration" / "simulator.json"
RANKS = 8
KINDS = ("all_gather", "reduce_scatter", "all_reduce", "all_to_all", "p2p")
SIZES = (1 << 20, 1 << 24, 1 << 28)
CONFIGS = {
    "system": "system.json",
    "remote_memory": "remote_memory.json",
    "network": "net_switch8.yml",
}
FINISHED = re.compile(r"sys\[(\d+)\] finished, (\d+) cycles")


def simulator() -> Path:
    root = os.environ.get("SIM_ROOT")
    binary = Path(root or "") / "src/astra-sim/build/astra_analytical/build/bin"
    binary = binary / "AstraSim_Analytical_Congestion_Aware"
    if not root or not binary.exists():
        print("set SIM_ROOT to the directory README.md installs into", file=sys.stderr)
        raise SystemExit(2)
    return binary


def workload(kind: str, group_size: int, size: int) -> Builder:
    """One *kind* on each group of *group_size* consecutive ranks (a pair for p2p)."""
    builder = Builder(RANKS)
    for rank in range(RANKS):
        group = [other for other in range(RANKS) if other // group_size == rank // group_size]
        if kind == "p2p":
            src, dst = group[0], group[1]
            if rank in (src, dst):
                builder.point("send" if rank == src else "recv", src, dst, "p2p", size, 0)
        else:
            builder.collective(rank, kind, kind, size, group)
    return builder


def simulate(binary: Path, builder: Builder, work: Path) -> int:
    """The time in ns the simulator takes for *builder*: the last rank to finish."""
    trace = builder.write(work / "trace")
    command = [
        str(binary),
        f"--workload-configuration={trace}",
        f"--comm-group-configuration={work / 'trace' / 'comm_group.json'}",
        f"--system-configuration={HERE / CONFIGS['system']}",
        f"--remote-memory-configuration={HERE / CONFIGS['remote_memory']}",
        f"--network-configuration={HERE / CONFIGS['network']}",
        f"--logging-folder={work / 'log'}",
    ]
    output = subprocess.run(command, capture_output=True, text=True, timeout=600).stdout
    finished = {int(rank): int(ns) for rank, ns in FINISHED.findall(output)}
    if len(finished) != RANKS:
        raise RuntimeError(f"{len(finished)} of {RANKS} ranks finished: {' '.join(command)}")
    return max(finished.values())


def main() -> int:
    binary = simulator()
    fits = []
    with tempfile.TemporaryDirectory() as tmp:
        for kind in KINDS:
            for group_size in (2,) if kind == "p2p" else (2, 4, 8):
                points = []
                for size in SIZES:
                    work = Path(tmp) / f"{kind}_{group_size}_{size}"
                    points.append((size, simulate(binary, workload(kind, group_size, size), work)))
                (x0, y0), (xm, ym), (x1, y1) = points
                slope = (y1 - y0) / (x1 - x0)
                base = y0 - slope * x0
                error = abs(base + slope * xm - ym) / ym
                fits.append(
                    {
                        "kind": kind,
                        "size": group_size,
                        "ns_at_zero": base,
                        "ns_per_byte": slope,
                        "points": points,
                        "middle_error": round(error, 6),
                    }
                )
                print(
                    f"{kind:15s} {group_size}  ns = {base:10.1f} + {slope:.6f} * bytes  middle error {error:.2%}"
                )
    record = {
        "simulator": "AstraSim_Analytical_Congestion_Aware",
        "configs": CONFIGS,
        "ranks": RANKS,
        "fit": "ns = ns_at_zero + ns_per_byte * size, through the smallest and the largest point",
        "fits": fits,
    }
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
