"""Render complete-training latency and device occupancy from saved evidence."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MODES = ("off", "on")
LABELS = ("Prefetch off", "Prefetch + async")
COLORS = ("#77808f", "#188678")
PHASES = dict(forward="#11875d", backward="#7652b7", optimizer="#d97706", nccl="#1676cb", other="#777777")
NAMES = dict(cp="PP + CP / DMuon", cp_ep="PP + CP + EP / DMuon", cp_hsdp="PP + CP / DMuon HSDP")


def comparison(data, output):
    cases = data["cases"]
    figure, axes = plt.subplots(2, len(cases), figsize=(5.3 * len(cases), 8), squeeze=False)
    states = ("Compute only", "Compute + NCCL", "NCCL only", "No observed kernel")
    colors = ("#11875d", "#d97706", "#1676cb", "#e5e7eb")
    for column, case in enumerate(cases):
        variants = [case["variants"][mode] for mode in MODES]
        top, bottom = axes[:, column]
        times = [v["seconds_per_step"] for v in variants]
        top.barh(range(2), times, color=COLORS, height=0.5)
        for row, v in enumerate(variants):
            top.scatter(v["samples_seconds_per_step"], [row] * len(v["samples_seconds_per_step"]), c="black", s=16, zorder=3)
            top.text(max(v["samples_seconds_per_step"]) + max(times) * 0.03, row,
                     f"{times[row]:.3f} s\n{v['valid_targets_per_second']:,.0f} targets/s\n{v['peak_allocated_bytes']/2**30:.1f} GiB peak", va="center", fontsize=9)
            metrics = v["metrics"]
            compute, nccl, overlap, span = (sum(m[key] for m in metrics) for key in ("compute_us", "nccl_us", "overlap_us", "device_span_us"))
            parts = (compute - overlap, overlap, nccl - overlap, span - compute - nccl + overlap)
            left = 0
            for amount, color in zip(parts, colors):
                width = 100 * amount / span
                bottom.barh(row, width, left=left, color=color, height=0.5)
                left += width
            bottom.text(0, row + 0.36, f"Overlap: {100*overlap/compute:.1f}% of compute; {100*overlap/nccl:.1f}% of NCCL", fontsize=8)
        top.set_xlim(0, max(max(v["samples_seconds_per_step"]) for v in variants) * 1.6)
        top.set_xlabel("Seconds / complete training step (unprofiled)")
        top.set_title(f"{NAMES[case['name']]}\nObserved off/on latency ratio: {times[0]/times[1]:.2f}×", loc="left", fontsize=12)
        bottom.set_xlim(0, 100)
        bottom.set_xlabel("% of captured device span, summed across ranks")
        bottom.set_title("Separate kernel capture", loc="left", fontsize=11)
        for axis in (top, bottom):
            axis.set_yticks(range(2), LABELS)
            axis.set_ylim(1.65, -0.65)
            axis.spines[["top", "right"]].set_visible(False)
    config = cases[0]["variants"]["off"]["config"]
    figure.suptitle(f"Qwen training: {config['shape']['layers']} layers, {config['tokens']:,} packed positions, "
                   f"{config['microbatches']} microbatches / data lane", x=0.05, ha="left", fontsize=17)
    figure.legend(handles=[Patch(color=color, label=label) for label, color in zip(states, colors)],
                  loc="lower center", bbox_to_anchor=(0.53, 0.08), ncol=4, frameon=False)
    figure.text(0.05, 0.02, "Paired policies share data, checkpointing, shapes and sharding. Dots: multi-step timing windows; see report for final-state drift.\n"
                "NCCL spans include waits. Unobserved kernel gaps can include host work and copies. Peak memory excludes the numerical reference and profiler.", fontsize=9)
    figure.tight_layout(rect=(0.01, 0.16, 1, 0.94))
    for extension in ("png", "svg"):
        figure.savefig(output / f"training_comparison.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def timelines(data, output):
    case = next((c for c in data["cases"] if c["name"] == "cp_ep"), data["cases"][0])
    figure, axes = plt.subplots(1, 2, figsize=(17, 8), sharex=True, sharey=True)
    ticks, labels = [], []
    for axis, mode, label in zip(axes, MODES, LABELS):
        for lane in case["variants"][mode]["first_step"]:
            for kind, intervals in lane["intervals"].items():
                row = lane["rank"] * 2 + (kind == "nccl")
                axis.broken_barh([(a, b-a) for a, b in intervals], (row-0.3, 0.6), color=PHASES[kind])
            if mode == MODES[0]:
                ticks.extend((lane["rank"]*2, lane["rank"]*2+1))
                stage = lane["rank"] // case["variants"][mode]["config"]["dp"]
                labels.extend((f"r{lane['rank']} s{stage} compute", f"r{lane['rank']} s{stage} NCCL"))
        axis.set_title(label, loc="left")
        axis.set_xlabel("Milliseconds from first kernel across ranks")
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(ticks, labels)
    axes[0].invert_yaxis()
    figure.suptitle(f"{NAMES[case['name']]}: first captured training step, all ranks", x=0.07, ha="left", fontsize=17)
    figure.legend(handles=[Patch(color=color, label=kind) for kind, color in PHASES.items()],
                  loc="lower center", bbox_to_anchor=(0.54, 0.065), ncol=5, frameon=False)
    figure.text(0.07, 0.02, "Each candidate has a separate origin; clocks are aligned across its ranks. Backward includes checkpoint recomputation.\n"
                "100 µs display grid; metrics retain exact timestamps. NCCL includes waits. Use unprofiled timing for speedups.", fontsize=9)
    figure.tight_layout(rect=(0.02, 0.13, 1, 0.94))
    for extension in ("png", "svg"):
        figure.savefig(output / f"training_timeline.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def operations(data, output, interactive=False):
    """Zoom a measured layer and expose named kernel groups for every rank."""
    case = next((c for c in data["cases"] if c["name"] == "cp_ep"), data["cases"][0])
    details = [case["variants"][mode]["layer_detail"] for mode in MODES]
    names = sorted(set().union(*details), key=lambda name: min(a for detail in details if name in detail for a, _ in detail[name]["intervals"]))
    figure, axes = plt.subplots(1, 2, figsize=(18, max(6, len(names)*0.38)), sharex=True, sharey=True)
    palette = dict(zip(names, plt.get_cmap("tab20").colors * 2))
    for axis, detail, label in zip(axes, details, LABELS):
        for row, name in enumerate(names):
            axis.broken_barh([(a, b-a) for a, b in detail.get(name, {}).get("intervals", [])], (row-0.3, 0.6), color=palette[name])
        axis.set_title(label, loc="left")
        axis.set_xlabel("Milliseconds from first kernel in this layer")
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    figure.suptitle(f"{NAMES[case['name']]}: layer 0, microbatch 0, rank 0", x=0.04, ha="left", fontsize=17)
    figure.text(0.04, 0.02, "Named operations come from launch correlation, thread-local scopes and actual CUDA/ATen operators.\n"
                "Scope-unassigned collectives retain their primitive name. 100 µs display grid; separate profiler capture.", fontsize=9)
    figure.tight_layout(rect=(0.01, 0.1, 1, 0.93))
    for extension in ("png", "svg"):
        figure.savefig(output / f"training_operations.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)
    if not interactive:
        return
    import plotly.graph_objects as go

    figure = go.Figure()
    choices = []
    for case in data["cases"]:
        for mode in MODES:
            for lane in case["variants"][mode]["first_step"]:
                begin = len(figure.data)
                labels = list(lane["operations"])
                for row, (name, intervals) in enumerate(lane["operations"].items()):
                    figure.add_trace(go.Bar(x=[b-a for a, b in intervals], base=[a for a, _ in intervals],
                                            y=[row]*len(intervals), orientation="h", width=0.6, name=name,
                                            visible=not choices, showlegend=False,
                                            hovertemplate=f"{name}<br>start %{{base:.3f}} ms<br>span %{{x:.3f}} ms<extra></extra>"))
                choices.append((f"{case['name']} / {mode} / rank {lane['rank']}", begin, len(figure.data), labels))
    figure.update_layout(template="plotly_white", height=1000, barmode="overlay", dragmode="zoom",
                         title="Named operations in a complete captured training step", xaxis_title="Milliseconds from first kernel across ranks",
                         margin=dict(l=340, t=120, b=80),
                         yaxis=dict(tickvals=list(range(len(choices[0][3]))), ticktext=choices[0][3], autorange="reversed"),
                         updatemenus=[dict(x=0, y=1.1, buttons=[dict(label=label, method="update",
                            args=[dict(visible=[a <= i < b for i in range(len(figure.data))]),
                                  {"yaxis.tickvals": list(range(len(labels))), "yaxis.ticktext": labels}]) for label, a, b, labels in choices])])
    figure.add_annotation(text="NCCL includes waits. Names describe observed operations; they do not prove fusion legality.",
                          xref="paper", yref="paper", x=0, y=-0.1, showarrow=False)
    figure.write_html(output / "training_operations.html", include_plotlyjs=True)


def pipeline_diagram(data, output):
    """Explain the executed dependency path and its checked communication boundaries."""
    from matplotlib.patches import FancyBboxPatch

    case = next((c for c in data["cases"] if c["name"] == "cp_ep"), None)
    if case is None:
        return
    run = case["variants"]["off"]
    config, shape = run["config"], run["config"]["shape"]
    tokens = config["tokens"] // config["cp"]
    kv = next(c for c in run["contract_examples"] if c["kind"] == "all_gather" and len(c["input_shape"]) == 5)
    payload = next(c for c in run["contract_examples"] if c["kind"] == "exchange" and c["dtype"] == "torch.float32")
    matrix = [[n // shape["hidden"] for n in row] for row in payload["counts"]]
    import math

    kv_mib = math.prod(kv["input_shape"])*4/2**20
    hidden_mib = tokens * shape["hidden"]*4/2**20
    forward = [
        "Parameter materialization\nDMuon owner broadcasts\n+ sharded-param all-gathers",
        f"Q/K/V projection GEMMs\nQ/K normalization + RoPE\n{tokens:,} local positions",
        f"CP K/V all-gather [C]\n{kv_mib:g} → {kv_mib*config['cp']:g} MiB\nparticipants {kv['participants']}",
        "SDPA attention\nlocal Q, global K/V\nthen output projection GEMM",
        f"Router GEMM + top-{shape['top_k']}\nEP count all-gather [C]\n{tokens*shape['top_k']:,} routes/rank",
        f"EP token + ID dispatch [C]\nuneven all-to-all\n{hidden_mib*shape['top_k']:g} MiB rows/rank",
        f"Local expert SwiGLU\ngate/up and down GEMMs\n{shape['experts']//config['ep']} experts/rank",
        f"EP return + weighted sum [C]\nnext layer, then PP send [C]\nhidden payload {hidden_mib:g} MiB",
    ]
    backward = [
        "Muon Newton–Schulz GEMMs\n+ momentum / AdamW update\nthen parameter publication",
        "Owner-gradient reduction\n+ sharded-param reduce-scatter\nreplica reduction when HSDP",
        "CP K/V grad reduce-scatter [C]\nthen projection backward GEMMs\ninput and weight gradients",
        "SDPA backward\nquery, key, value gradients\n+ output-projection backward",
        "EP dispatch backward [C]\nreverse token exchange\nmerge router/residual gradients",
        "Expert backward GEMMs\ninput and weight gradients\nper expert owner",
        "EP return backward [C]\nreverse output exchange\ntoken gradients to experts",
        "PP gradient receive [C]\nweighted-combine backward\nrecompute checkpointed stage",
    ]
    figure, axis = plt.subplots(figsize=(21, 7))
    axis.set_xlim(-0.1, 8.1)
    axis.set_ylim(-0.8, 3.2)
    axis.axis("off")
    for row, nodes in enumerate((forward, backward)):
        y = 2.1 if row == 0 else 0.5
        for i, label in enumerate(nodes):
            color = "#e0edf9" if "[C]" in label else "#e4f1e9"
            axis.add_patch(FancyBboxPatch((i+0.02, y), 0.92, 0.7, boxstyle="round,pad=0.025", facecolor=color, edgecolor="#678"))
            axis.text(i+0.48, y+0.35, label, ha="center", va="center", fontsize=9)
            if i < 7:
                start, end = ((i+0.96, y+0.35), (i+1.01, y+0.35)) if row == 0 else ((i+1.01, y+0.35), (i+0.96, y+0.35))
                axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#345"))
    axis.text(0, 2.98, "FORWARD →", fontsize=12, weight="bold")
    axis.text(0, 1.42, "← BACKWARD / UPDATE", fontsize=12, weight="bold")
    axis.text(0, -0.08, f"One checked EP dispatch: token counts {matrix}; participants {payload['participants']}. Counts change with routing.", fontsize=11)
    axis.text(0, -0.42, "[C] Shape/participant/count contracts checked by TileFoundry; PP endpoints also match in FIFO order. Compute executes in PyTorch.\n"
              "Arrows show the main dependency path; residual/router branches are omitted. Future-layer prefetch still requires readiness waits.\n"
              "The full training compute DAG and its physical lifetimes are not yet lowered into TileFoundry; the diagram does not claim that proof.", fontsize=10)
    figure.suptitle("PP + CP + EP / DMuon: operations, dependencies and checked boundaries", x=0.035, ha="left", fontsize=18)
    figure.tight_layout(rect=(0.01, 0, 1, 0.94))
    for extension in ("png", "svg"):
        figure.savefig(output / f"training_pipeline.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    with (gzip.open if args.evidence.suffix == ".gz" else open)(args.evidence, "rt") as source:
        data = json.load(source)
    args.output.mkdir(parents=True, exist_ok=True)
    comparison(data, args.output)
    timelines(data, args.output)
    operations(data, args.output, args.interactive)
    pipeline_diagram(data, args.output)


if __name__ == "__main__":
    main()
