"""Render measured gradient-fusion evidence without running CUDA or a simulator."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

MODES = ("serial", "pipelined", "fused", "fused_pipelined")
LABELS = ("Serial", "Pipelined", "Fused", "Fused + pipeline")
COLORS = ("#77808f", "#3274b5", "#bb6d1f", "#188678")
PHASES = dict(accumulate="#3274b5", collective="#7853a0", normalize="#d78a31")


def size_label(case):
    amount = case["payload_bytes"] / 2**20
    return f"{amount:g} MiB" if amount >= 1 else f"{case['payload_bytes'] / 1024:g} KiB"


def load_evidence(path):
    data = json.loads(path.read_text())
    if data.get("schema") != 1 or data.get("status") != "passed" or not data.get("cases"):
        raise ValueError("expected a completed gradient-fusion evidence report")
    for case in data["cases"]:
        for mode in MODES:
            variant = case["variants"][mode]
            if variant["numerical_check"] != "passed" or variant["legality"] != "checked":
                raise ValueError("every displayed candidate must pass numerical and core checks")
            if len(variant["samples_us"]) != data["rounds"] or len(variant["profiles"]) != data["world"]:
                raise ValueError("timing rounds or rank profiles are missing")
    return data


def save(figure, directory, name):
    for extension in ("png", "svg"):
        figure.savefig(directory / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(figure)


def comparison(data, directory):
    cases = data["cases"]
    figure = plt.figure(figsize=(15, 9), facecolor="white")
    grid = figure.add_gridspec(2, max(3, len(cases)), hspace=0.55, wspace=0.50)
    figure.subplots_adjust(bottom=0.21, top=0.88, left=0.12, right=0.98)
    figure.suptitle("Gradient buckets: scheduling and fusion measured separately", fontsize=18, x=0.06, ha="left")
    for index, case in enumerate(cases):
        axes = figure.add_subplot(grid[0, index])
        variants = [case["variants"][mode] for mode in MODES]
        values = [variant["median_us"] for variant in variants]
        intervals = [np.percentile(variant["samples_us"], [10, 90]) for variant in variants]
        error = np.array([[value - span[0], span[1] - value] for value, span in zip(values, intervals)]).T
        axes.barh(range(4), values, xerr=error, color=COLORS, height=0.6, capsize=3, error_kw=dict(elinewidth=1))
        for row, value in enumerate(values):
            axes.text(intervals[row][1] + max(values) * 0.025, row, f"{value:.0f} µs", va="center", fontsize=9)
        axes.set_yticks(range(4), LABELS)
        axes.invert_yaxis()
        axes.set_xlim(0, max(span[1] for span in intervals) * 1.28)
        axes.set_title(f"{size_label(case)} / rank\nUnprofiled latency", loc="left", fontsize=12)
        axes.set_xlabel("µs per complete bucket")
        axes.text(0, -0.26, f"Fused + pipeline: {values[0] / values[3]:.2f}× vs serial", transform=axes.transAxes, fontsize=10, fontweight="bold")
        axes.spines[["top", "right"]].set_visible(False)
    case = cases[-1]
    variants = [case["variants"][mode] for mode in MODES]
    axes = figure.add_subplot(grid[1, 0])
    values = [variant["workspace_bytes"] / 2**20 for variant in variants]
    axes.barh(range(4), values, color=COLORS, height=0.6)
    for row, value in enumerate(values):
        axes.text(value + max(values) * 0.03, row, f"{value:g}", va="center")
    axes.set_yticks(range(4), LABELS)
    axes.invert_yaxis()
    axes.set_xlim(0, max(values) * 1.22)
    axes.set_xlabel("MiB of explicit workspace / rank")
    axes.set_title(f"{size_label(case)}: core allocation proof", loc="left", fontsize=12)
    axes.text(0, -0.28, "Inputs, outputs and NCCL internal buffers excluded", transform=axes.transAxes, fontsize=8)
    axes.spines[["top", "right"]].set_visible(False)
    axes = figure.add_subplot(grid[1, 1])
    values = [variant["local_kernel_bytes"] / case["payload_bytes"] for variant in variants]
    axes.barh(range(4), values, color=COLORS, height=0.6)
    for row, value in enumerate(values):
        axes.text(value + 0.1, row, f"{value:g}×", va="center")
    axes.set_yticks(range(4), LABELS)
    axes.invert_yaxis()
    axes.set_xlim(0, max(values) * 1.25)
    axes.set_xlabel("Local kernel bytes / bucket bytes")
    axes.set_title("Core access proof: 40% fewer bytes", loc="left", fontsize=12)
    axes.text(0, -0.28, "Read + write footprints; NCCL traffic excluded", transform=axes.transAxes, fontsize=8)
    axes.spines[["top", "right"]].set_visible(False)
    axes = figure.add_subplot(grid[1, 2])
    for row, variant in enumerate(variants):
        profiles = variant["profiles"]
        points = [profile["nccl_overlap_fraction"] * 100 for profile in profiles]
        weighted = 100 * sum(p["overlap_us"] for p in profiles) / sum(p["nccl_us"] for p in profiles)
        axes.barh(row, weighted, color=COLORS[row], alpha=0.3, height=0.6)
        axes.scatter(points, row + np.linspace(-0.18, 0.18, len(points)), color=COLORS[row], s=20, zorder=3)
        axes.text(max(points) + 2, row, f"{weighted:.1f}%", va="center", fontsize=9)
    axes.set_yticks(range(4), LABELS)
    axes.invert_yaxis()
    axes.set_xlim(0, 100)
    axes.set_xlabel("% of NCCL kernel time overlapping compute")
    axes.set_title("Separate profiler capture", loc="left", fontsize=12)
    axes.text(0, -0.28, "Dots: ranks; bars: NCCL-time-weighted fraction", transform=axes.transAxes, fontsize=8)
    axes.spines[["top", "right"]].set_visible(False)
    figure.text(0.06, 0.035, f"{data['rounds']} randomized rounds × {data['iterations']} complete calls; median with P10–P90 observed spread. "
                f"{data['steps']} fresh-input checks against FP64 passed.\n"
                "NCCL intervals include waits. Profiler overlap is not a latency prediction or wire utilization. Isolated gradient boundary, not full training.", fontsize=9)
    save(figure, directory, "comparison")


def timelines(data, directory):
    case = data["cases"][-1]
    figure, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=True)
    maximum = 0
    for row, mode in enumerate(MODES):
        profile = case["variants"][mode]["profiles"][0]
        timeline = profile["first_step"]
        maximum = max(maximum, *(event["start_us"] + event["duration_us"] for event in timeline))
        counts = {}
        for event in timeline:
            phase = event["phase"]
            lane = 0 if phase != "collective" else 1
            axes[row].broken_barh([(event["start_us"], event["duration_us"])], (lane - 0.28, 0.56), facecolors=PHASES[phase])
            counts[phase] = counts.get(phase, 0) + 1
        axes[row].set_yticks([0, 1], ["Compute", "NCCL"])
        axes[row].set_ylim(1.6, -0.8)
        axes[row].set_title(f"{LABELS[row]} — compute/NCCL intersection {profile['overlap_us']:.1f} µs/step", loc="left", fontsize=11)
        axes[row].grid(axis="x", alpha=0.2)
        axes[row].spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlim(0, maximum * 1.03)
    axes[-1].set_xlabel("µs from first device kernel in each independently captured step")
    figure.suptitle(f"{size_label(case)} bucket: measured device intervals, rank 0", x=0.07, ha="left", fontsize=17)
    figure.legend(handles=[Patch(color=color, label=phase.capitalize()) for phase, color in PHASES.items()],
                  loc="lower center", bbox_to_anchor=(0.52, 0.055), ncol=3, frameon=False)
    figure.text(0.07, 0.02, "First profiled step per variant; title metrics average all profiled steps. Separate captures do not share an origin.\n"
                "Profiler timing is diagnostic; use the unprofiled comparison for speedups. NCCL includes rendezvous and waits.", fontsize=9)
    figure.tight_layout(rect=(0.02, 0.12, 1, 0.95))
    save(figure, directory, "timeline")


def interactive(data, directory):
    """A standalone zoomable timeline with every captured rank and payload."""
    import plotly.graph_objects as go

    figure = go.Figure()
    groups = []
    for case in data["cases"]:
        for rank in range(data["world"]):
            start = len(figure.data)
            for row, mode in enumerate(MODES):
                events = case["variants"][mode]["profiles"][rank]["first_step"]
                for phase, color in PHASES.items():
                    selected = [event for event in events if event["phase"] == phase]
                    lane = row * 2 + (phase == "collective")
                    figure.add_trace(go.Bar(x=[event["duration_us"] for event in selected],
                                            base=[event["start_us"] for event in selected], y=[lane] * len(selected),
                                            orientation="h", width=0.55, name=phase, legendgroup=phase,
                                            marker_color=color, visible=False, showlegend=row == 0,
                                            hovertemplate=f"{mode}: {phase}<br>start %{{base:.2f}} µs<br>duration %{{x:.2f}} µs<extra></extra>"))
            groups.append((f"{size_label(case)}, rank {rank}", start, len(figure.data)))
    for trace in figure.data[:groups[0][2]]:
        trace.visible = True
    figure.update_layout(template="plotly_white", height=700, barmode="overlay",
                         title="Measured gradient-bucket device intervals (each variant starts at zero)",
                         xaxis_title="µs from first GPU kernel", dragmode="zoom",
                         yaxis=dict(tickvals=list(range(8)), ticktext=[f"{label} / {lane}" for label in LABELS for lane in ("compute", "NCCL")], autorange="reversed"),
                         margin=dict(l=195, t=115, b=70), legend=dict(orientation="h"),
                         updatemenus=[dict(x=0, y=1.13, buttons=[dict(label=label, method="update", args=[dict(visible=[start <= index < end for index in range(len(figure.data))])])
                                                                for label, start, end in groups])])
    figure.add_annotation(text="Profiler diagnostic; NCCL includes waits. Latency evidence is measured separately without profiling.",
                          xref="paper", yref="paper", x=0, y=-0.13, showarrow=False)
    figure.write_html(directory / "timeline.html", include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    data = load_evidence(args.evidence)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.titlesize": 12})
    comparison(data, args.output)
    timelines(data, args.output)
    if args.interactive:
        interactive(data, args.output)


if __name__ == "__main__":
    main()
