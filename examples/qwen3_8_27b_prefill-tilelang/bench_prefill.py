"""SGLang prefill latency against prompt length.

    bench_prefill.py --ckpt DIR [--lens 1024,4096,8192] [--out reports/x.json]

**What is measured.** One request, `max_new_tokens=1`. The engine therefore
does one prefill, samples once, and stops: `meta_info.e2e_latency` is the time
to first token. Three things make that number mean "prefill":

  radix cache off     every repeat redoes the prefill. With it on the second
                      repeat hits the cache and reports ~0, which is how a
                      baseline silently becomes two orders of magnitude wrong.
                      `cached_tokens == 0` is asserted on every single run,
                      not sampled.

  lengths <= 8192     8192 is `chunked_prefill_size` on an H200 (sglang derives
                      it from GPU memory, `arg_groups/memory_hook.py::
                      _handle_gpu_memory_settings`). At 8192 the admission test is
                      `extend_input_len <= rem_chunk_tokens`, which holds, so
                      the prompt goes through in **one forward**. 8193 would be
                      split in two. So these three lengths compare one forward
                      against one forward.

  overhead measured   a 1-token prompt is timed the same way. That run is all
                      fixed cost -- scheduling, sampling, detokenize, IPC --
                      and is reported beside every row so a reader can see how
                      much of the latency is not prefill.

**Repeats.** `--warm 2` untimed then `--reps 7` timed, median reported, spread
reported. The warmup is not optional: sglang captures a CUDA graph per prefill
shape and the first replay of a shape is far slower than the rest -- without it
8192 reads 20% slow. A different token id per repeat, so even a cache left on
by accident could not hit.
"""
import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import flops as flopmod

H200_BF16_PEAK_TFLOPS = 989.0   # NVIDIA's figure, BF16 dense, at boost
H200_BOOST_MHZ = 1980.0         # the clock that figure assumes


def sm_clock_mhz() -> float:
    """The SM clock right now. Clocks cannot be locked on this machine (no
    permission for `nvidia-smi -lgc`), and prefill is a sustained full-load
    workload, so the clock that was actually reached is reported with the
    number rather than assumed."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", dev, "--query-gpu=clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lens", default="1024,4096,8192")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunked-prefill-size", type=int, default=8192)
    ap.add_argument("--label", default="default")
    ap.add_argument("--log-level", default="warning",
                    help="pass 'info' to see the scheduler's own `Prefill batch. #new-token:` lines, which say whether a prompt was split across forwards")
    args = ap.parse_args()

    # the engine talks to itself over loopback; a SOCKS proxy in the environment
    # breaks that, and the failure looks like a hang.
    for k in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
              "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(k, None)

    import sglang as sgl

    ckpt = Path(args.ckpt)
    lens = [int(x) for x in args.lens.split(",")]
    out_path = Path(args.out) if args.out else (
        Path(__file__).parent / "results" / f"prefill-{args.label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for n in lens:
        assert n <= args.chunked_prefill_size, (
            f"len {n} is above chunked_prefill_size {args.chunked_prefill_size}; "
            "it would be split across forwards and the row would not mean "
            "'one prefill'")

    print(f"### sglang {sgl.__version__}  ckpt {ckpt}  lens {lens}", flush=True)
    llm = sgl.Engine(
        model_path=str(ckpt),
        trust_remote_code=True,
        tp_size=args.tp,
        disable_radix_cache=True,
        chunked_prefill_size=args.chunked_prefill_size,
        log_level=args.log_level,
    )

    def run(n_ctx: int, tok: int):
        sp = {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True}
        o = llm.generate(input_ids=[[tok] * n_ctx], sampling_params=sp)
        m = o[0]["meta_info"]
        assert m["cached_tokens"] == 0, (
            f"n={n_ctx} hit the prefix cache ({m['cached_tokens']} tokens); "
            "this run did not prefill and the number is meaningless")
        assert m["completion_tokens"] == 1, (
            f"n={n_ctx} produced {m['completion_tokens']} tokens, expected 1")
        return m["e2e_latency"]

    for _ in range(3):                       # warm kernels, autotune, allocator
        run(512, 7)

    def timed(n_ctx: int):
        # Per-shape warmup. sglang captures a CUDA graph per prefill shape
        # (0.5.21 does this for prefill, not only decode), and the first replay
        # of a shape is not representative of the rest.
        for k in range(args.warm):
            run(n_ctx, 101 + k)
        got = []
        clk = []
        for k in range(args.reps):
            got.append(run(n_ctx, 11 + k))
            clk.append(sm_clock_mhz())     # read while the GPU is still hot
        return statistics.median(got), min(got), max(got), statistics.median(clk)

    # The serving path's idle wake-up, measured at the smallest captured graph
    # shape. **This is not subtracted from any row.** A 1-token prompt pads 4x
    # to that bucket, past _MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2, so it
    # falls back to eager and times a different path -- 4 is the smallest
    # length that still runs the graph.
    #
    # It also does not add at the lengths that matter: the 4096 -> 8192 slope
    # extrapolates to an intercept of about zero. The scheduler loop sleeps
    # when idle and is already spinning when a forward is in flight, so this
    # cost is paid only by a request that arrives at an idle engine.
    wake_s, _, _, _ = timed(4)
    print(f"idle wake-up (n=4, smallest graph shape): {wake_s*1e3:.2f} ms "
          f"-- disclosed, not subtracted", flush=True)

    rows = []
    for n in lens:
        med, lo, hi, clk = timed(n)
        f = flopmod.prefill_flops(ckpt, n)
        tflops = f["total"] / med / 1e12
        # Clocks cannot be locked here, and this card ran at ~1500 MHz under
        # sustained prefill load rather than the 1980 MHz the 989 TFLOP/s
        # figure assumes. Both denominators are reported: the nominal one is
        # comparable with published numbers, the clock-adjusted one says how
        # much of the silicon that was actually available got used.
        peak_at_clock = H200_BF16_PEAK_TFLOPS * clk / H200_BOOST_MHZ
        rows.append({
            "n": n,
            "latency_ms": med * 1e3,
            "spread_ms": (hi - lo) * 1e3,
            "idle_wakeup_ms": wake_s * 1e3,
            "prefill_tok_s": n / med,
            "flop": f["total"],
            "tflop_s": tflops,
            "pct_peak_nominal": 100 * tflops / H200_BF16_PEAK_TFLOPS,
            "pct_peak_at_clock": 100 * tflops / peak_at_clock,
            "peak_at_clock_tflops": peak_at_clock,
            "roofline_ms": f["total"] / (H200_BF16_PEAK_TFLOPS * 1e12) * 1e3,
            "reps": args.reps,
            "sm_clock_mhz": clk,
        })
        r = rows[-1]
        print(f"n={n:>6}  {r['latency_ms']:9.2f} ms  "
              f"{r['prefill_tok_s']:9.0f} tok/s  "
              f"{r['tflop_s']:7.1f} TFLOP/s  "
              f"{r['pct_peak_nominal']:5.1f}% nom / "
              f"{r['pct_peak_at_clock']:5.1f}% at-clock  "
              f"(floor {r['roofline_ms']:.1f} ms, spread {r['spread_ms']:.2f} ms, "
              f"sm {r['sm_clock_mhz']:.0f} MHz)",
              flush=True)
        json.dump({
            "sglang": sgl.__version__,
            "ckpt": str(ckpt),
            "label": args.label,
            "tp": args.tp,
            "chunked_prefill_size": args.chunked_prefill_size,
            "disable_radix_cache": True,
            "peak_tflops": H200_BF16_PEAK_TFLOPS,
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": rows,
        }, open(out_path, "w"), indent=1)

    llm.shutdown()
    print(f"### done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
