"""Summarize measured CUDA overlap and export a shared-clock training timeline."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
from math import ceil, floor, isfinite
from pathlib import Path


def union(intervals):
    """Merge overlapping device intervals without summing concurrent streams twice."""
    merged = []
    for start, end in sorted(intervals):
        if not isfinite(start) or not isfinite(end) or end < start:
            raise ValueError("interval endpoints must be finite and ordered")
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        elif end > start:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def intersection(left, right):
    """Measure time when both sets have at least one active CUDA kernel."""
    left, right = union(left), union(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def display_intervals(intervals, shift):
    """Coalesce a 100-microsecond display grid; metrics retain exact intervals."""
    return union((floor((start + shift) / 100) / 10, ceil((end + shift) / 100) / 10)
                 for start, end in intervals)


def read_rank(folder, rank):
    """Correlate kernels, including CUDA graph replay, to host launch ranges."""
    manifest = json.loads((folder / f"rank{rank}.json").read_text())
    with gzip.open(folder / f"profile-rank{rank}.json.gz", "rt") as handle:
        trace = json.load(handle)
    events = trace["traceEvents"]
    annotations = [e for e in events if e.get("cat") == "user_annotation" and e.get("ph") == "X"]
    windows = [(e["ts"], e["ts"] + e["dur"]) for e in annotations if e["name"] == "training/window"]
    if len(windows) != 1:
        raise ValueError("expected one recorded training window")
    phases = sorted((e for e in annotations if e["name"].startswith("pipeline/")
                     or e["name"] == "training/optimizer"), key=lambda e: e["ts"])
    starts = [e["ts"] for e in phases]
    launches = {e.get("args", {}).get("correlation"): e for e in events
                if e.get("cat") in {"cuda_runtime", "cuda_driver"} and "correlation" in e.get("args", {})}
    kernels, missing = [], 0
    for event in events:
        if event.get("cat") != "kernel":
            continue
        launch = launches.get(event.get("args", {}).get("correlation"))
        if launch is None:
            missing += 1
            continue
        if not windows[0][0] <= launch["ts"] <= windows[0][1]:
            continue
        index = bisect_right(starts, launch["ts"]) - 1
        phase = "other"
        if index >= 0 and launch["ts"] < phases[index]["ts"] + phases[index]["dur"]:
            name = phases[index]["name"]
            phase = "optimizer" if name == "training/optimizer" else name.split("/")[1]
        kind = "nccl" if "nccl" in event["name"].lower() else phase
        kernels.append(dict(event, kind=kind, phase=phase))
    if missing or not kernels:
        raise ValueError(f"incomplete CUDA trace: {missing} uncorrelated kernels, {len(kernels)} attributed")
    by_kind = defaultdict(list)
    for event in kernels:
        by_kind[event["kind"]].append((event["ts"], event["ts"] + event["dur"]))
    communication = by_kind["nccl"]
    compute = [interval for kind, intervals in by_kind.items() if kind != "nccl" for interval in intervals]
    overlap = intersection(compute, communication)
    communication_us = duration(communication)
    metrics = dict(rank=rank, stage=manifest["stage"], data_rank=manifest["data_rank"],
                   compute_us=duration(compute), nccl_us=communication_us, overlap_us=overlap,
                   nccl_without_compute_us=communication_us - overlap,
                   nccl_overlap_fraction=overlap / communication_us if communication_us else 0,
                   compute_overlap_fraction=overlap / duration(compute) if duration(compute) else 0,
                   overlap_by_phase_us={kind: intersection(intervals, communication)
                                        for kind, intervals in by_kind.items() if kind != "nccl"},
                   nccl_kernels=dict(Counter(e["name"] for e in kernels if e["kind"] == "nccl")),
                   kernel_count=len(kernels))
    return dict(manifest=manifest, base=trace["baseTimeNanoseconds"], host=trace["host_name"],
                metrics=metrics, kernels=kernels, phases=phases, intervals=by_kind,
                annotations=[e for e in annotations if windows[0][0] <= e["ts"] <= windows[0][1]
                             and e["name"].startswith(("pipeline/", "qwen/layer/", "training/step/", "training/optimizer"))])


def write_timeline(folder, ranks):
    """Write an interactive GPU interval plot and a detailed Perfetto-compatible trace."""
    import plotly.graph_objects as go
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if len({rank["host"] for rank in ranks}) != 1:
        raise ValueError("multi-host clock alignment has not been established")
    base = min(rank["base"] for rank in ranks)
    origin = min(e["ts"] + (rank["base"] - base) / 1000 for rank in ranks for e in rank["kernels"])
    colors = dict(forward="#11875d", backward="#7652b7", optimizer="#d97706", nccl="#1676cb", other="#777777")
    figure = go.Figure()
    preview, axes = plt.subplots(figsize=(15, max(6, len(ranks))))
    merged, shown = [], set()
    for data in ranks:
        rank = data["metrics"]["rank"]
        shift = (data["base"] - base) / 1000 - origin
        for kind, intervals in data["intervals"].items():
            x, y = [], []
            lane = rank * 2 + (kind == "nccl")
            displayed = display_intervals(intervals, shift)
            for start, end in displayed:
                x.extend((start, end, None))
                y.extend((lane, lane, None))
            axes.broken_barh([(start, end - start) for start, end in displayed],
                             (lane - 0.3, 0.6), color=colors[kind])
            figure.add_trace(go.Scattergl(x=x, y=y, mode="lines", name=kind, legendgroup=kind,
                                          showlegend=kind not in shown, line=dict(color=colors[kind], width=9),
                                          hovertemplate=f"rank {rank}; {kind}; %{{x:.2f}} ms<extra></extra>"))
            shown.add(kind)
        merged.append(dict(ph="M", pid=rank, tid=0, name="process_name",
                           args=dict(name=f"rank {rank} GPU, stage {data['metrics']['stage']}")))
        merged.append(dict(ph="M", pid=1000 + rank, tid=0, name="process_name",
                           args=dict(name=f"rank {rank} host ranges")))
        for event in data["kernels"]:
            merged.append(dict(ph="X", cat=event["kind"], name=event["name"], pid=rank, tid=event["tid"],
                               ts=event["ts"] + shift, dur=event["dur"], args=event["args"] | dict(phase=event["phase"])))
        for event in data["annotations"]:
            lane = 1 if event["name"].startswith("qwen/") else 2 if event["name"].startswith("training/step/") else 0
            merged.append(dict(ph="X", cat="host phase", name=event["name"], pid=1000 + rank, tid=lane,
                               ts=event["ts"] + shift, dur=event["dur"]))
    ticks = [(data["metrics"]["rank"] * 2 + row,
              f"r{data['metrics']['rank']} / stage {data['metrics']['stage']} / {'NCCL' if row else 'compute'}")
             for data in ranks for row in (0, 1)]
    figure.update_layout(title=f"{folder.name}: measured GPU kernels (union across streams)",
                         xaxis_title="Milliseconds from first training kernel", template="plotly_white",
                         height=max(500, len(ranks) * 90), dragmode="zoom", hovermode="closest",
                         yaxis=dict(tickmode="array", tickvals=[x for x, _ in ticks],
                                    ticktext=[label for _, label in ticks], autorange="reversed"),
                         legend=dict(orientation="h"), margin=dict(l=180, b=100))
    note = "100 µs display grid; exact intervals in Perfetto/JSON. NCCL includes waits; overlap is not wire time."
    figure.add_annotation(text=note,
                           xref="paper", yref="paper", x=0, y=-0.12, showarrow=False)
    figure.write_html(folder / "timeline.html", include_plotlyjs=True)
    axes.set_yticks([x for x, _ in ticks], [label for _, label in ticks])
    axes.invert_yaxis()
    axes.set_xlabel("Milliseconds from first training kernel")
    axes.set_title(f"{folder.name}: measured training kernels")
    axes.legend(handles=[Line2D([0], [0], color=color, lw=6, label=kind)
                          for kind, color in colors.items()], loc="upper center",
                 bbox_to_anchor=(0.5, -0.09), ncol=len(colors), frameon=False)
    preview.text(0.5, 0.01, note, ha="center", fontsize=9)
    preview.tight_layout(rect=(0, 0.06, 1, 1))
    preview.savefig(folder / "timeline.png", dpi=160)
    plt.close(preview)
    with gzip.open(folder / "timeline.json.gz", "wt") as handle:
        json.dump(dict(traceEvents=merged, displayTimeUnit="ms"), handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("rank*.json"), key=lambda p: int(p.stem[4:]))
    ranks = [read_rank(args.directory, int(path.stem[4:])) for path in paths]
    if not ranks:
        parser.error("no rank manifests found")
    manifest = ranks[0]["manifest"]
    if [rank["metrics"]["rank"] for rank in ranks] != list(range(manifest["pp"] * manifest["dp"])):
        parser.error("incomplete rank manifests")
    for rank in ranks:
        other = rank["manifest"]
        if other["status"] != "passed" or not other.get("profile"):
            parser.error("every rank must have passed with a training profile")
        for key in ("shape", "pp", "dp", "microbatches", "tokens", "optimizer", "overlap", "boundary", "transport"):
            if manifest[key] != other[key]:
                parser.error(f"rank manifests disagree on {key}")
    write_timeline(args.directory, ranks)
    total_communication = sum(data["metrics"]["nccl_us"] for data in ranks)
    summary = dict(scope="CUDA kernel intervals on one host; reference operations excluded by launch correlation",
                   nccl_wait_caveat="NCCL kernels include rendezvous and spinning, not only data movement",
                   rank_time_weighted_nccl_overlap=(sum(data["metrics"]["overlap_us"] for data in ranks)
                                                    / total_communication if total_communication else 0),
                   unprofiled_seconds_per_step=ranks[0]["manifest"]["profile"]["unprofiled_seconds_per_step"],
                   ranks=[data["metrics"] for data in ranks])
    (args.directory / "overlap.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "ranks"}, indent=2))


if __name__ == "__main__":
    main()
