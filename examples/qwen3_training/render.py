"""Draw timelines as a PNG: per rank, the compute, communication and memory lanes on one time axis.

    uv run --no-project --with matplotlib==3.9.2 python -m examples.qwen3_training.render
        runs/cp_hsdp/none/program/timeline.chrome.json [more timelines ...] --out step.png
        [--ranks 0,4] [--from <ns> --to <ns>]

Each timeline given is one panel, drawn above the next, so two runs of the same
rank compare on one time axis (D51). Only ``timeline.chrome.json`` is read: this
file does not import ``tilefoundry``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# The validated categorical order (dataviz reference palette), one slot per kind, never cycled.
KIND_COLOR = {
    "all_gather": "#2a78d6",
    "reduce_scatter": "#eb6834",
    "all_reduce": "#1baf7a",
    "all_to_all": "#eda100",
    "p2p": "#e87ba4",
}
COMPUTE = "#898781"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
CAPACITY = "#d03b3b"
SURFACE = "#fcfcfb"


def kind_of(event: dict) -> str:
    kind = event.get("args", {}).get("kind", "compute")
    return "p2p" if kind in ("send", "recv") else kind


def load(path: Path, ranks, start: float | None, end: float | None) -> dict:
    """The bars, memory steps and peak of each chosen rank of one timeline, in ms."""
    trace = json.loads(path.read_text())
    lo = -float("inf") if start is None else start / 1e6
    hi = float("inf") if end is None else end / 1e6
    panel = {"meta": trace.get("otherData", {}), "ranks": {}}
    for event in trace["traceEvents"]:
        rank = event.get("pid")
        if ranks is not None and rank not in ranks:
            continue
        lane = panel["ranks"].setdefault(
            rank, {"label": f"rank {rank}", "bars": {}, "memory": [], "peak": None}
        )
        phase = event.get("ph")
        if phase == "M" and event["name"] == "process_name":
            lane["label"] = event["args"]["name"]
        elif phase == "X":
            begin, finish = event["ts"] / 1000, (event["ts"] + event["dur"]) / 1000
            if finish < lo or begin > hi:
                continue
            begin, finish = max(begin, lo), min(finish, hi)
            lane["bars"].setdefault(kind_of(event), []).append((begin, finish - begin))
        elif phase == "C" and event["name"] == "memory":
            lane["memory"].append(
                (event["ts"] / 1000, event["args"]["used"], event["args"]["capacity"])
            )
        elif phase == "i":
            lane["peak"] = (event["ts"] / 1000, event["args"]["bytes"], event["name"])
    for lane in panel["ranks"].values():
        lane["memory"].sort()
    return panel


def title_of(panel: dict) -> str:
    meta = panel["meta"]
    title = f"{meta.get('config')} / {meta.get('recompute')} / {meta.get('policy')}"
    return f"{title} / {meta['tokens']} tokens" if meta.get("tokens") else title


def draw(panels: list[dict], out: Path, start: float | None, end: float | None) -> None:
    rows = [(panel, rank) for panel in panels for rank in sorted(panel["ranks"])]
    figure, axes = plt.subplots(
        2 * len(rows),
        1,
        figsize=(14, height := 1.9 * len(rows) + 1.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.9] * len(rows), "hspace": 0.45},
        facecolor=SURFACE,
    )
    kinds_seen = set()
    for index, (panel, rank) in enumerate(rows):
        lane = panel["ranks"][rank]
        bars_axis, memory_axis = axes[2 * index], axes[2 * index + 1]
        for axis in (bars_axis, memory_axis):
            axis.set_facecolor(SURFACE)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            axis.spines["left"].set_color(GRID)
            axis.spines["bottom"].set_color(GRID)
            axis.tick_params(colors=MUTED, labelsize=8)
        for kind, bars in lane["bars"].items():
            y = (1.1, 0.8) if kind == "compute" else (0.1, 0.8)
            color = COMPUTE if kind == "compute" else KIND_COLOR.get(kind, MUTED)
            bars_axis.broken_barh(bars, y, facecolors=color, linewidth=0)
            kinds_seen.add(kind)
        bars_axis.set_yticks([0.5, 1.5], ["communication", "compute"], fontsize=8, color=MUTED)
        bars_axis.set_ylim(0, 2)
        bars_axis.set_title(
            f"{title_of(panel)}  ·  {lane['label']}",
            fontsize=9,
            color=INK,
            loc="left",
        )
        if lane["memory"]:
            times = [time for time, _, _ in lane["memory"]]
            used = [value / 1e9 for _, value, _ in lane["memory"]]
            capacity = lane["memory"][0][2] / 1e9
            memory_axis.step(times, used, where="post", color=INK, linewidth=1.2)
            memory_axis.axhline(capacity, color=CAPACITY, linewidth=1.2, linestyle="--")
            memory_axis.text(
                1.0,
                capacity,
                f" capacity {capacity:.0f} GB",
                transform=memory_axis.get_yaxis_transform(),
                fontsize=8,
                color=MUTED,
                va="center",
                ha="left",
            )
            top = max(max(used), capacity)
            memory_axis.set_ylim(0, top * 1.15)
        if lane["peak"] is not None:
            time, value, name = lane["peak"]
            if (start is None or time >= start / 1e6) and (end is None or time <= end / 1e6):
                memory_axis.plot(time, value / 1e9, marker="v", markersize=8, color=INK)
                label = "OOM" if name == "OOM" else "peak"
                memory_axis.annotate(
                    f"{label} {value / 1e9:.1f} GB",
                    (time, value / 1e9),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                    color=INK,
                )
        memory_axis.set_ylabel("GB", fontsize=8, color=MUTED)
        memory_axis.grid(axis="y", color=GRID, linewidth=0.6)
    axes[-1].set_xlabel("time (ms)", fontsize=9, color=MUTED)
    if start is not None or end is not None:
        axes[-1].set_xlim(
            None if start is None else start / 1e6, None if end is None else end / 1e6
        )
    handles = [Patch(color=COMPUTE, label="compute")]
    handles += [
        Patch(color=color, label=kind.replace("_", " "))
        for kind, color in KIND_COLOR.items()
        if kind in kinds_seen
    ]
    handles += [
        Line2D([], [], color=INK, linewidth=1.2, label="memory used"),
        Line2D([], [], color=CAPACITY, linewidth=1.2, linestyle="--", label="capacity"),
    ]
    meta = panels[0]["meta"]
    # Margins in inches, so a tall figure has no empty band above or below its lanes.
    figure.subplots_adjust(top=1 - 0.75 / height, bottom=0.7 / height)
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 0.05 / height),
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )
    figure.text(
        0.5,
        0.1 / height,
        f"communication: {meta.get('communication', '')}  ·  compute: {meta.get('compute', '')}",
        ha="center",
        fontsize=7,
        color=MUTED,
    )
    figure.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("timelines", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", help="comma-separated ranks; all when left out")
    parser.add_argument("--from", dest="start", type=float, help="start of the window, ns")
    parser.add_argument("--to", dest="end", type=float, help="end of the window, ns")
    args = parser.parse_args(argv)
    ranks = None if args.ranks is None else {int(rank) for rank in args.ranks.split(",")}
    panels = [load(path, ranks, args.start, args.end) for path in args.timelines]
    draw(panels, args.out, args.start, args.end)
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
