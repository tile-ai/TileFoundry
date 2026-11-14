"""Summarize training and render measured or modeled operation pipelines as SVG."""

import argparse
import gzip
import json
import math
from pathlib import Path

from tilefoundry import utils

plt = utils.LazyModule("matplotlib.pyplot")
patches = utils.LazyModule("matplotlib.patches")

MATCH = ("shape", "pp", "dp", "cp", "ep", "packed", "tokens", "microbatches", "dtype", "optimizer",
         "activation_checkpointing", "boundary", "hsdp_replicas", "ns_backend", "forward_unshard",
         "routing_check", "transport", "total_training_steps", "sources", "attention", "attention_dtype", "flash_attn", "numerical_policy")


def read_manifests(folder):
    """Require one completed, consistent manifest for every declared rank."""
    paths = sorted(folder.glob("rank*.json"), key=lambda p: int(p.stem[4:]))
    manifests = [json.loads(path.read_text()) for path in paths]
    if not manifests or any(m["status"] != "passed" or not m.get("profile") for m in manifests):
        raise ValueError(f"{folder}: every rank must finish validation and profiling")
    first = manifests[0]
    world = first["pp"] * first["dp"]
    if [m["rank"] for m in manifests] != list(range(world)):
        raise ValueError("rank manifests are incomplete")
    config = {key: first.get(key) for key in MATCH}
    for m in manifests:
        if any(m.get(key) != config[key] for key in MATCH):
            raise ValueError("rank manifests disagree on the workload")
    return manifests


def summarize(folder):
    """Compare timing and numerical checks; detailed capture analysis is a separate command."""
    records = read_manifests(folder)
    first = records[0]
    profile = first["profile"]
    examples = {}
    for contract in first["cp_ep_contracts"]:
        key = contract["kind"], tuple(contract["participants"]), contract["dtype"], len(contract["input_shape"])
        examples.setdefault(key, contract)
    return dict(config={key:first.get(key) for key in MATCH}, overlap=first["overlap"], profile_steps=profile["profiled_steps"],
                seconds_per_step=profile["unprofiled_seconds_per_step"],
                samples_seconds_per_step=profile["samples_seconds_per_step"],
                rank_window_seconds=profile["rank_window_seconds"],
                warmup_steps=profile["warmup_steps"], steps_per_round=profile["steps_per_round"],
                valid_targets_per_step=profile["valid_targets_per_step"],
                valid_targets_per_second=profile["valid_targets_per_second"],
                peak_allocated_bytes=max(m["profile"]["training_peak_allocated_bytes"] for m in records),
                numerical_steps=len(first["steps"]),
                max_gradient_error=max(s["max_gradient_error"] for m in records for s in m["steps"]),
                max_gradient_relative_l2=max((s["max_gradient_relative_l2"] for m in records for s in m["steps"]
                                              if "max_gradient_relative_l2" in s), default=None),
                changed_routing_sets=sum(s["changed_sets"] for m in records for s in m["routing_comparisons"]),
                matched_transfers=first["matched_transfers"],
                cp_ep_contracts=sorted({c["kind"] for m in records for c in m["cp_ep_contracts"]}),
                contract_examples=list(examples.values()),
                pp_examples=[e for e in first["transfers"] if e["step"] == 0 and e["microbatch"] == 0],
                software={k: first[k] for k in ("torch", "transformers", "nccl", "dmuon_version")})

def check_pair(left, right):
    if left["config"] != right["config"]:
        raise ValueError("paired candidates must use the same model, data and sharding configuration")
    for key in ("valid_targets_per_step", "warmup_steps", "steps_per_round", "profile_steps", "numerical_steps"):
        if left[key] != right[key]:
            raise ValueError(f"paired candidates disagree on {key}")
    if len(left["samples_seconds_per_step"]) != len(right["samples_seconds_per_step"]):
        raise ValueError("paired candidates need identical timing windows")

def weight_difference(left, right):
    """Report drift without assuming bitwise deterministic training or relaxing the oracle."""
    count, different, maximum, squared, baseline, samples = 0, 0, 0.0, 0.0, 0.0, 0
    for a, b in zip(left, right):
        # Published weights are replicated; count one copy per pipeline stage.
        if a["stage_local_rank"]:
            continue
        for name, fingerprint in a["final_weight_sha256"].items():
            count += 1
            different += fingerprint != b["final_weight_sha256"][name]
            for x, y in zip(a["final_weight_samples"][name], b["final_weight_samples"][name]):
                maximum = max(maximum, abs(x-y))
                squared += (x-y)**2
                baseline += x*x
                samples += 1
    return dict(parameter_tensors=count, different_hashes=different, sampled_elements=samples,
                sample_max_abs=maximum, sample_relative_l2=math.sqrt(squared / baseline) if baseline else None,
                scope="all-tensor hashes and finite checks; fixed samples diagnose drift, not full-weight equivalence")

def profile_run(folder, *, all_ranks=False):
    """Retain individual device activities and host scopes for an inspectable capture."""
    from .trace import read_rank

    records = read_manifests(folder)
    ranks = [read_rank(folder, m["rank"]) for m in records if all_ranks or m["stage_local_rank"] == 0]
    if not ranks or len({r["host"] for r in ranks}) != 1:
        raise ValueError("profile export requires completed ranks on an aligned host clock")
    clock = min(r["base"] for r in ranks)
    origin = min(e["ts"] + (r["base"]-clock)/1000 for r in ranks for e in r["kernels"])
    output = []
    for rank in ranks:
        shift = (rank["base"]-clock)/1000-origin
        events = []
        for e in rank["kernels"]:
            attributes = e.get("args", {})
            details = {key: attributes[key] for key in ("grid", "block", "registers per thread", "shared memory",
                        "est. achieved occupancy %", "correlation", "graph id") if key in attributes}
            events.append(dict(ts=e["ts"]+shift, dur=e["dur"], stream=str(attributes.get("stream", e["tid"])),
                               name=e["name"], operation=e["operation"], phase=e["phase"], step=e["step"],
                               microbatch=e["microbatch"], layer=e["layer"], kind=e["kind"],
                               operator=e["operator"], launch_delay_us=e["ts"]-e["launch_ts"], details=details))
        for e in rank["copies"]:
            events.append(dict(ts=e["ts"]+shift, dur=e["dur"], stream=str(e.get("args", {}).get("stream", e["tid"])),
                               name=e["name"], operation=e["name"], phase=e["phase"], kind="copy",
                               step=e["step"], microbatch=e["microbatch"], layer=e["layer"],
                               details={k:v for k,v in e.get("args", {}).items() if k in ("bytes", "memory bandwidth (GB/s)")}))
        scopes = [dict(ts=e["ts"]+shift, dur=e["dur"], name=e["name"], stream="CPU scopes", operation=e["name"],
                       phase="host", kind="host") for e in rank["annotations"] if e["name"].startswith(("pipeline/", "training/optimizer"))]
        memory = [dict(ts=e["ts"]+shift, allocated=e["args"].get("Total Allocated", 0),
                       reserved=e["args"].get("Total Reserved", 0)) for e in rank["memory"]]
        memory.sort(key=lambda e: e["ts"])
        manifest = rank["manifest"]
        output.append(dict(rank=manifest["rank"], stage=manifest["stage"], events=events, scopes=scopes,
                           memory=memory, metrics=rank["metrics"],
                           training_peak_bytes=manifest["profile"]["training_peak_allocated_bytes"]))
    first = ranks[0]["manifest"]
    source = folder / "source_sha256.json"
    return dict(name=folder.name, mode="measured", ranks=output, source_run=str(folder),
                source_sha256=json.loads(source.read_text()) if source.exists() else None,
                config={k:first.get(k) for k in ("shape", "cp", "ep", "pp", "hsdp_replicas", "overlap", "tokens", "microbatches", "dtype",
                                                 "attention", "attention_dtype", "numerical_policy")},
                software={k:first.get(k) for k in ("torch", "transformers", "dmuon_version", "attention", "flash_attn")},
                timing=("All ranks. " if all_ranks else "One representative rank per pipeline stage. ")+
                       "CUDA activities attributed by launch scope; NCCL includes readiness and protocol waits.")



# Analysis consumes the same activity records that the diagram displays.
def diagnose(run):
    """Report observed idle intervals and lags; only modeled edges establish causes."""
    from bisect import bisect_left, bisect_right
    from collections import defaultdict
    from statistics import median

    from .trace import duration, union

    modeled = run["mode"] == "modeled"
    all_events = [e for rank in run["ranks"] for e in rank["events"]]
    by_key = {e["details"]["key"]: e for e in all_events if "key" in e.get("details", {})}
    bounds = {}
    for event in all_events:
        step = event.get("step")
        if step is not None:
            lo, hi = bounds.get(step, (float("inf"), 0))
            bounds[step] = min(lo, event["ts"]), max(hi, event["ts"]+event["dur"])
    findings, ranks = [], []
    for rank in run["ranks"]:
        events = sorted((e for e in rank["events"] if e["dur"] > 0), key=lambda e: e["ts"])
        idle = 0
        for step, (start, stop) in sorted(bounds.items()):
            selected = [e for e in events if e["ts"] < stop and e["ts"]+e["dur"] > start]
            starts = [e["ts"] for e in selected]
            ended = sorted(selected, key=lambda e:e["ts"]+e["dur"])
            ends = [e["ts"]+e["dur"] for e in ended]
            occupied = union((max(start,e["ts"]), min(stop,e["ts"]+e["dur"])) for e in selected)
            cursor = start
            for begin, end in [*occupied, (stop, stop)]:
                if begin > cursor:
                    idle += begin-cursor
                    next_index, previous_index = bisect_left(starts, begin), bisect_right(ends, cursor)-1
                    following = selected[next_index] if next_index<len(selected) else None
                    preceding = ended[previous_index] if previous_index>=0 else None
                    cause, blocker = "unattributed", None
                    if modeled and following:
                        detail = following["details"]
                        key = detail.get("critical_predecessor")
                        blocker = by_key.get(key)
                        if blocker and blocker["ts"]+blocker["dur"] >= begin-1e-3:
                            cause = "resource" if detail["resource_wait_us"] > 0 else "dependency"
                    findings.append(dict(kind="stage_idle", cause=cause, rank=rank["rank"], stage=rank["stage"],
                                         step=step, start_us=cursor, duration_us=begin-cursor,
                                         before=preceding["operation"] if preceding else None,
                                         after=following["operation"] if following else None,
                                         blocker=blocker["details"]["key"] if blocker else None,
                                         key=following.get("details", {}).get("key") if following else None,
                                         blocked_by=blocker["operation"] if blocker else None,
                                         blocker_stage=blocker.get("details", {}).get("stage") if blocker else None,
                                         critical_path=following.get("details", {}).get("critical_path", False) if following else False))
                cursor = max(cursor, end)
        ranks.append(dict(rank=rank["rank"], stage=rank["stage"], idle_us=idle,
                          window_us=sum(hi-lo for lo,hi in bounds.values())))
        if modeled:
            for event in events:
                detail = event["details"]
                if detail["resource_wait_us"] > 1e-3:
                    blocker = by_key.get(detail["critical_predecessor"])
                    findings.append(dict(kind="resource_wait", cause="resource", rank=rank["rank"], stage=rank["stage"],
                                         step=event["step"], start_us=detail["ready_us"], duration_us=detail["resource_wait_us"],
                                         operation=event["operation"], key=detail["key"],
                                         blocker=detail["critical_predecessor"], blocked_by=blocker["operation"] if blocker else None,
                                         resources=detail.get("resource_predecessors", {}), critical_path=detail["critical_path"]))
        else:
            groups = defaultdict(lambda: defaultdict(list))
            for event in events:
                if event.get("microbatch") is not None:
                    groups[event["operation"], event["phase"], event.get("layer")][event["step"], event["microbatch"]].append(event)
            for (operation, phase, layer), instances in groups.items():
                if len(instances) < 4:
                    continue
                samples = {key: duration((e["ts"], e["ts"]+e["dur"]) for e in value) for key,value in instances.items()}
                baseline = median(samples.values())
                mad = median(abs(value-baseline) for value in samples.values())
                for key,value in samples.items():
                    if value > max(1.5*baseline, baseline+3*mad, baseline+100):
                        findings.append(dict(kind="operation_lag", cause="unattributed", rank=rank["rank"], stage=rank["stage"],
                                             step=key[0], microbatch=key[1], layer=layer, phase=phase, operation=operation,
                                             start_us=min(e["ts"] for e in instances[key]), duration_us=value,
                                             median_us=baseline, excess_us=value-baseline, sample_count=len(samples)))
    counts = {kind:sum(f["kind"] == kind for f in findings) for kind in ("stage_idle", "resource_wait", "operation_lag")}
    # Keep evidence bounded; prioritize critical waits before their duration.
    largest = {kind:sorted((f for f in findings if f["kind"] == kind),
                           key=lambda f:(f.get("critical_path",False),f.get("excess_us", f["duration_us"])), reverse=True)[:20] for kind in counts}
    return dict(mode=run["mode"], ranks=ranks, counts=counts, findings=largest,
                interpretation=("Dependency/resource causes follow the authored schedule, not a physical execution proof." if modeled else
                                "Idle means no captured device activity. NCCL residency includes waits. Causes require further tracing."),
                lag_rule="Same rank, operation, phase and layer; >=4 microbatches; >1.5x median, >median+3 MAD and >median+100us.")


# Row labels describe work/ownership boundaries, independently of its transport.
LANES = (
    ("Projection / norm / RoPE", "projection", "q/k/v", "pointwise", "normalization"),
    ("CP K/V gather / gradient scatter", "cp "),
    ("Attention forward / gradients", "attention"),
    ("Router / EP dispatch / return", "ep ", "router", "routing"),
    ("Expert MLP / gradients", "expert"),
    ("Embedding / LM head / loss", "embedding", "lm head", "loss"),
    ("PP activation / gradient transfer", "pp ", "grouped p2p"),
    ("Parameter materialization / publication", "publication", "materializ", "unshard"),
    ("Owner-gradient reduction", "owner-gradient", "dmuon gradient"),
    ("Muon / AdamW update", "muon", "adamw", "momentum"),
    ("Backward GEMMs (unassigned layer)", "backward gemm"),
    ("Unassigned NCCL", "nccl"),
    ("Device copies / fills", "memcpy", "memset"),
)
PHASE_COLORS = {"forward":"#277da8", "recompute":"#a06ab4", "backward":"#db8544", "optimizer":"#36866a", "other":"#8995a4"}


def lane(event):
    name = event["operation"].lower().removeprefix("recompute: ")
    if name.startswith(("muon ", "momentum", "adamw")):
        return "Muon / AdamW update"
    return next((label for label,*words in LANES if any(word in name for word in words)), "Other attributed CUDA")


def phase(event):
    replay_exchange = (event["phase"] == "backward" and event.get("layer") is not None
                       and event["operation"] in ("CP K/V all-gather", "EP count all-gather",
                                                   "EP token/ID dispatch", "EP output return"))
    if event["operation"].startswith("Recompute: ") or replay_exchange:
        return "recompute"
    return event["phase"] if event["phase"] in PHASE_COLORS else "other"


def figure_identity(run):
    """Use one configuration name for the document, SVG title and output filename."""
    config = run["config"]
    topology = config.get("grid",config)
    parts = [axis.upper() for axis in ("pp","cp","ep") if topology[axis] > 1]
    parts.append("DMuon HSDP" if config.get("hsdp_replicas",1) > 1 else "DMuon")
    title = " + ".join(parts)
    stem = "-".join(parts).lower().replace(" ","-")
    if run["mode"] == "modeled":
        title = f"Modeled {title} ({topology['ranks']:,} ranks)"
        stem = f"modeled-{stem}-{topology['ranks']}-ranks"
    else:
        stem += f"-prefetch-{config['overlap']}"
    return title,stem


def draw_operations(axis, ranks, origin, scale, *, highlight=False):
    from .trace import union

    visible = [e for r in ranks for e in r["events"] if e["dur"] > 0]
    resolution = (max(e["ts"]+e["dur"] for e in visible)-min(e["ts"] for e in visible))/1400 if visible else 0
    labels, positions, y = [], {}, 0
    for rank in ranks:
        prefix = f"Pipeline stage {rank['stage']+1}"
        if sum(r["stage"] == rank["stage"] for r in ranks) > 1:
            prefix += f" / rank {rank['rank']}"
        events = [e for e in rank["events"] if e["dur"] > 0]
        actions = {}
        for event in events:
            if event.get("microbatch") is not None and event["phase"] in ("forward","backward"):
                key = event["phase"][0].upper()+str(event["microbatch"])
                actions[key] = min(actions.get(key,float("inf")),event["ts"])
        if actions:
            labels.append((y,f"{prefix} · Microbatch starts (F / B)"))
            span = max(e["ts"]+e["dur"] for e in events)-min(e["ts"] for e in events)
            last_text = -float("inf")
            for label,when in sorted(actions.items(),key=lambda pair:pair[1]):
                axis.plot([(when-origin)/scale]*2,[y-.1,y+.25],color="#526579",lw=.6)
                if when-last_text > span*.018:
                    axis.text((when-origin)/scale,y-.15,label,fontsize=6.5,va="bottom")
                    last_text = when
            y += 1
        names = [label for label,*_ in LANES if any(lane(e)==label for e in events)]
        if any(lane(e)=="Other attributed CUDA" for e in events):
            names.append("Other attributed CUDA")
        for name in names:
            labels.append((y, f"{prefix} · {name}"))
            positions[rank["rank"],name] = y
            for kind,color in PHASE_COLORS.items():
                spans = union((e["ts"],e["ts"]+e["dur"]) for e in events if lane(e)==name and phase(e)==kind)
                if spans:
                    displayed = []
                    for start,end in spans:
                        if displayed and start-displayed[-1][1] < resolution:
                            displayed[-1] = displayed[-1][0],end
                        else:
                            displayed.append((start,end))
                    axis.broken_barh([((a-origin)/scale,(b-a)/scale) for a,b in displayed], (y-.31,.62),
                                     facecolors=color, linewidth=0)
            if highlight:
                critical = [(e["ts"],e["ts"]+e["dur"]) for e in events
                            if lane(e)==name and e.get("details",{}).get("critical_path")]
                if critical:
                    axis.broken_barh([((a-origin)/scale,(b-a)/scale) for a,b in union(critical)],
                                     (y-.34,.68),facecolors="none",edgecolors="#a33f3f",linewidth=.7)
            y += 1
        axis.axhline(y-.25,color="#cbd5df",lw=.6)
        y += .7
    axis.set_yticks([row for row,_ in labels],[label for _,label in labels],fontsize=8)
    axis.set_ylim(y-.7,-.8)
    axis.spines[["top","right"]].set_visible(False)
    axis.grid(axis="x",alpha=.15)
    return positions


def save_svg(figure, path, **kwargs):
    """Keep generated SVGs deterministic and free of trailing whitespace."""
    figure.savefig(path, metadata={"Date": None, "Creator": "TileFoundry"}, **kwargs)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def pipeline_diagram(run, path, *, step=None):
    """An operation pipeline and a diagnostic zoom; no synthetic semantic cartoon."""
    available = {e["step"] for r in run["ranks"] for e in r["events"] if e.get("step") is not None}
    if not available:
        raise ValueError("no completed step to display")
    step = min(available) if step is None else step
    ranks = [dict(rank=r["rank"],stage=r["stage"],events=[e for e in r["events"] if e.get("step")==step]) for r in run["ranks"]]
    if any(not r["events"] for r in ranks):
        raise ValueError(f"step {step} is not present on every displayed rank")
    events = [e for r in ranks for e in r["events"]]
    origin, end = min(e["ts"] for e in events), max(e["ts"]+e["dur"] for e in events)
    scale,unit = (1e6,"s") if end-origin>=1e6 else (1000,"ms")
    diagnostics = diagnose(dict(run,ranks=ranks))
    candidates = sorted(diagnostics["findings"]["resource_wait"] + diagnostics["findings"]["stage_idle"],
                        key=lambda f:(f.get("critical_path",False),f["duration_us"]),reverse=True)
    focus = next((f for f in candidates if f.get("blocked_by")), candidates[0] if candidates else None)
    # A fixed-size first-microbatch view is useful even when no idle interval exists.
    first = ranks[0]
    detail = [e for e in first["events"] if e["phase"]=="forward" and str(e.get("microbatch"))=="0"]
    if focus:
        middle = focus["start_us"] + focus["duration_us"]/2
        width = max(focus["duration_us"]*1.4, (end-origin)*.04)
        lo,hi = max(origin,middle-width/2), min(end,middle+width/2)
    elif detail:
        lo,hi = min(e["ts"] for e in detail), max(e["ts"]+e["dur"] for e in detail)
    else:
        lo,hi = origin,end
    zoom_ranks = [dict(r,events=[dict(e,ts=max(lo,e["ts"]),dur=min(hi,e["ts"]+e["dur"])-max(lo,e["ts"]))
                               for e in r["events"] if e["ts"]<hi and e["ts"]+e["dur"]>lo]) for r in ranks]
    rows = sum(1+len({lane(e) for e in r["events"] if e["dur"]>0}) for r in ranks)
    zoom_rows = sum(1+len({lane(e) for e in r["events"] if e["dur"]>0}) for r in zoom_ranks)
    plt.rcParams.update({"svg.fonttype":"none","svg.hashsalt":"qwen-training","font.size":9})
    height = 4+rows*.23+zoom_rows*.20
    figure,axes = plt.subplots(2,1,figsize=(19,height),
                               gridspec_kw=dict(height_ratios=(max(3,rows*.23),max(2,zoom_rows*.20)),hspace=2.2/(height-2.5)))
    draw_operations(axes[0],ranks,origin,scale)
    axes[0].set(xlim=(0,(end-origin)/scale),xlabel=f"{unit} from first device activity · common origin across stages")
    axes[0].set_title(f"Complete step {step} · named operation intervals, checkpoint replay, and optimizer boundary",loc="left",fontsize=12)
    axes[0].axvspan((lo-origin)/scale,(hi-origin)/scale,color="#e5c44d",alpha=.14)
    positions = draw_operations(axes[1],zoom_ranks,origin,scale,highlight=True)
    keyed = {e["details"]["key"]:(r["rank"],e) for r in ranks for e in r["events"] if "key" in e.get("details",{})}
    if focus and focus.get("blocker") in keyed and focus.get("key") in keyed:
        source_rank,source = keyed[focus["blocker"]]
        target_rank,target = keyed[focus["key"]]
        if (source_rank,lane(source)) in positions and (target_rank,lane(target)) in positions:
            x = (max(lo,source["ts"])+min(hi,source["ts"]+source["dur"]))/2
            axes[1].annotate("",xy=((target["ts"]-origin)/scale,positions[target_rank,lane(target)]),
                             xytext=((x-origin)/scale,positions[source_rank,lane(source)]),
                             arrowprops=dict(arrowstyle="->",color="#a33f3f",lw=1.3,linestyle="--",connectionstyle="arc3,rad=.2"))
    axes[1].set(xlim=((lo-origin)/scale,(hi-origin)/scale),xlabel=f"{unit} from the same step origin · highlighted window above")
    if focus:
        reason = f"{focus['cause']} · {focus['duration_us']/1000:.2f} ms · Pipeline stage {focus['stage']+1}"
        if run["mode"] == "modeled":
            reason += " · on critical path" if focus.get("critical_path") else " · off critical path"
        successor = focus.get("operation",focus.get("after")) or "step end"
        if focus.get("blocked_by"):
            reason += f"\n{successor} waits for {focus['blocked_by']}"
        else:
            reason += f"\nNo device activity between {focus.get('before') or 'step start'} and {successor}; cause not established"
    else:
        reason = "First forward microbatch · no idle interval detected"
    axes[1].set_title("Diagnostic window: "+reason,loc="left",fontsize=11,pad=12)
    config = run["config"]
    shape,topology = config.get("shape",config),config.get("grid",config)
    subtitle = f"{shape['layers']} layers · {config['tokens']:,} packed positions · PP={topology['pp']} CP={topology['cp']} EP={topology['ep']}"
    if run["mode"] == "measured":
        subtitle += f" · replicas={config['hsdp_replicas']} · prefetch {config['overlap']}"
    title,_ = figure_identity(run)
    figure.suptitle(f"{title}\n{run['mode'].capitalize()} operation pipeline · {subtitle}",x=.025,y=1-.15/height,ha="left",fontsize=16)
    figure.legend(handles=[patches.Patch(color=color,label="Unassigned phase" if name=="other" else name.capitalize()) for name,color in PHASE_COLORS.items()],
                  loc="lower center",bbox_to_anchor=(.55,.65/height),ncol=5,frameon=False)
    note = ("Inferred durations; red outlines mark the modeled critical path; dashed edge identifies the blocker. No numerical or memory-capacity proof."
            if run["mode"]=="modeled" else "Measured CUDA intervals; NCCL includes readiness/protocol waits. Idle gaps and operation lags do not establish their causes.")
    if run["mode"] == "measured" and len(ranks) == topology["pp"]:
        note = "One rank per pipeline stage shown. "+note
    figure.text(.025,.15/height,note+"\nDisplay merges subpixel gaps within an operation lane; diagnostic records use original timestamps.",fontsize=9)
    figure.subplots_adjust(left=.31,right=.985,top=1-1.05/height,bottom=1.55/height)
    save_svg(figure,path,bbox_inches="tight")
    plt.close(figure)


def profile_artifacts(runs, output):
    """One data source for diagnostic records and reproducible SVG pipelines."""
    output.mkdir(parents=True,exist_ok=True)
    data = dict(schema=2,runs=runs,timestamp_unit="us")
    (output/"profiles.json.gz").write_bytes(gzip.compress(json.dumps(data,separators=(",",":")).encode(),mtime=0))
    (output/"diagnostics.json").write_text(json.dumps({run["name"]:diagnose(run) for run in runs},indent=2)+"\n")
    for run in runs:
        _,name = figure_identity(run)
        pipeline_diagram(run,output/f"{name}.svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs",type=Path,nargs="+",help="completed training capture directories")
    source.add_argument("--profiles",type=Path,help="saved profiles.json.gz; no CUDA required")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--all-ranks",action="store_true",help="include every rank instead of one per pipeline stage")
    args = parser.parse_args()
    if args.runs:
        runs = [profile_run(path,all_ranks=args.all_ranks) for path in args.runs]
    else:
        with gzip.open(args.profiles,"rt") as source:
            runs = json.load(source)["runs"]
    profile_artifacts(runs,args.output)


if __name__ == "__main__":
    main()
