"""Render a checked gradient boundary as measured, chunk-resolved operation pipelines."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tilefoundry import utils

from ..report import save_svg
from .program import MODES

plt = utils.LazyModule("matplotlib.pyplot")
patches = utils.LazyModule("matplotlib.patches")
LABELS = ("Serial", "Pipelined", "Reduction + normalization fused", "Fused + pipelined")
COLORS = dict(accumulate="#277da8", collective="#9464af", normalize="#36866a")


def load_evidence(path):
    data = json.loads(path.read_text())
    if data.get("schema") != 1 or data.get("status") != "passed" or not data.get("cases"):
        raise ValueError("expected a completed gradient-fusion evidence report")
    for case in data["cases"]:
        for mode in MODES:
            variant = case["variants"][mode]
            if variant["numerical_check"] != "passed" or variant["legality"] != "checked":
                raise ValueError("every candidate must pass numerical and core checks")
            if len(variant["samples_us"]) != data["rounds"] or len(variant["profiles"]) != data["world"]:
                raise ValueError("timing rounds or rank profiles are missing")
    return data


def pipeline(data, path):
    case = data["cases"][-1]
    plt.rcParams.update({"svg.fonttype":"none", "svg.hashsalt":"gradient-fusion", "font.size":10})
    figure,axes = plt.subplots(4,1,figsize=(16,12),sharex=True)
    maximum = 0
    for axis,mode,label in zip(axes,MODES,LABELS):
        variant = case["variants"][mode]
        events = variant["profiles"][0]["first_step"]
        counts,previous = defaultdict(int),{}
        for event in sorted(events,key=lambda e:e["start_us"]):
            phase = event["phase"]
            chunk = counts[phase]
            counts[phase] += 1
            start,end = event["start_us"],event["start_us"]+event["duration_us"]
            maximum = max(maximum,end)
            axis.broken_barh([(start,end-start)],(chunk-.28,.56),facecolors=COLORS[phase])
            if chunk in previous:
                axis.annotate("",xy=(start,chunk),xytext=(previous[chunk],chunk),
                              arrowprops=dict(arrowstyle="->",lw=.8,color="#445467"))
            previous[chunk] = end
        expected = dict(accumulate=case["chunks"],collective=case["chunks"])
        if not mode.startswith("fused"):
            expected["normalize"] = case["chunks"]
        if counts != expected:
            raise ValueError("cannot associate captured operations with the authored chunk order")
        axis.set_yticks(range(case["chunks"]),[f"Chunk {i}" for i in range(case["chunks"])])
        axis.set_ylim(case["chunks"]-.4,-.7)
        reduction = "NCCL average → output" if mode.startswith("fused") else "NCCL sum → multiply by 1/ranks"
        axis.set_title(f"{label}: a + b → {reduction}\n"
                       f"Core: {variant['workspace_bytes']/2**20:g} MiB scratch · {variant['local_kernel_launches']} local launches · "
                       f"{len(variant['overlap_candidates'])} independent occurrence pairs · "
                       f"{variant['local_kernel_bytes']/case['payload_bytes']:g}× payload in local accesses\n"
                       f"Unprofiled latency: {variant['median_us']:.1f} µs median · "
                       f"{min(variant['samples_us']):.1f}–{max(variant['samples_us']):.1f} µs across {data['rounds']} rounds",loc="left",fontsize=10)
        axis.grid(axis="x",alpha=.15)
        axis.spines[["top","right"]].set_visible(False)
    axes[-1].set(xlim=(0,maximum*1.015),xlabel="µs from the first GPU activity in each variant's capture")
    figure.suptitle(f"Gradient fusion\n{case['payload_bytes']/2**20:g} MiB/rank · {case['chunks']} chunks · "
                    "checked completion order and measured execution",x=.07,ha="left",fontsize=17)
    names = dict(accumulate="Accumulate microbatch gradients: a + b", collective="NCCL sum / average",normalize="Normalize reduced gradient: × 1/ranks")
    figure.legend(handles=[patches.Patch(color=color,label=names[phase]) for phase,color in COLORS.items()],
                  loc="lower center",bbox_to_anchor=(.51,.065),ncol=3,frameon=False)
    figure.text(.07,.02,"Arrows follow the checked per-chunk producer/consumer order. Missing normalize operations are absorbed into NCCL average.\n"
                "Rank 0, first diagnostic step. NCCL residency includes waits; use separately measured timing rounds for latency comparisons.\n"
                "Scratch excludes caller inputs/outputs and NCCL internal allocations. This is a gradient boundary, not a fused training implementation.",fontsize=9)
    figure.tight_layout(rect=(.03,.12,.99,.93),h_pad=2)
    save_svg(figure,path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    data = load_evidence(args.evidence)
    args.output.mkdir(parents=True,exist_ok=True)
    pipeline(data,args.output/"gradient-fusion.svg")


if __name__ == "__main__":
    main()
