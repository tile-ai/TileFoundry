"""SGLang decode tok/s against context length.

    bench_sglang.py <config json> <contexts, comma separated> <output json>

**The method, and why it is this one.** Two earlier versions produced numbers
that are not physically possible -- 701 tok/s at 4096, which is faster than the
roofline floor, and half of them negative. The cause was the radix cache: the
second time the same prompt arrives its prefill is nearly free, so subtracting
one total from another subtracts noise rather than prefill.

Turning the cache off is what makes the two runs symmetric:

    radix cache off  ->  run(n, 1) and run(n, 1+S) both certainly redo prefill
    decode = (t(1+S) - t(1)) / S     the subtraction leaves S decode steps

**Why off rather than deliberately hit.** Letting the prefix hit and having
sglang skip prefill entirely was the first idea, but `mamba_track_interval=256`
means an SSM checkpoint exists only every 256 tokens: below that there is nothing
to reuse and `cached_tokens` stays 0, and in between whether it hits depends on
alignment. So prefill was sometimes subtracted and sometimes not.

Turning the cache off does not weaken the baseline -- the radix cache only
affects prefill reuse, and decode speed does not depend on it. What is compared
here is pure decode on both sides, so the baseline has to be pure decode too:
not "with prefill subtracted", but with prefill not run at all. Times come from
`meta_info.e2e_latency` (measured server side); three runs, median.

**An assertion that fails throws instead of reporting** -- a baseline that is
quietly wrong by two orders of magnitude is worse than no baseline.
"""
import json, os, statistics, sys, time
import paths


def main():
    os.environ.pop("ALL_PROXY", None); os.environ.pop("all_proxy", None)
    import sglang as sgl
    CKPT = str(paths.need("ckpt"))
    cfg = json.loads(sys.argv[1]); ctxs = [int(x) for x in sys.argv[2].split(",")]
    out_path = sys.argv[3]
    print(f"### config {cfg}", flush=True)
    llm = sgl.Engine(model_path=CKPT, trust_remote_code=True, tp_size=1,
                     context_length=262144, **cfg)

    def run(n_ctx, n_new, tok):
        sp = {"max_new_tokens": n_new, "temperature": 0.0, "ignore_eos": True}
        o = llm.generate(input_ids=[[tok] * n_ctx], sampling_params=sp)
        m = o[0]["meta_info"]
        return m["e2e_latency"], m["cached_tokens"], m["completion_tokens"]

    run(512, 8, 5); run(512, 8, 5)                       # warm the cuda graphs
    rows = []
    ttft = float("nan")
    for i, n in enumerate(ctxs):
        tok = 11 + i                                     # a different token per
        S = 256 if n < 131072 else 32                    # length: 262080+33 < 262144
        got = []
        for k in range(3):
            # **Neither run hits the cache** (the radix cache is off), so what the
            # subtraction leaves is S decode steps: prefill, scheduling and
            # sampling are fixed costs and cancel.
            t1, c1, g1 = run(n, 1, tok)
            t2, c2, g2 = run(n, 1 + S, tok)
            assert c1 == 0 and c2 == 0, f"ctx={n} hit the cache ({c1}/{c2}); the subtraction does not hold"
            assert g1 == 1 and g2 == 1 + S, f"ctx={n} wrong step counts {g1} {g2}"
            got.append((t2 - t1) / S)
            if k == 0:
                ttft = t1
        ms = statistics.median(got) * 1e3
        assert ms > 0, f"ctx={n} subtraction gave {ms:.3f} ms, which is impossible"
        rows.append({"ctx": n, "ttft_ms": ttft * 1e3, "decode_tps": 1e3 / ms,
                     "ms_per_token": ms, "steps": S,
                     "spread_ms": (max(got) - min(got)) * 1e3})
        print(f"ctx={n:>7}  TTFT {ttft*1e3:9.1f} ms   decode {1e3/ms:7.2f} tok/s"
              f"   {ms:7.3f} ms/token   ({S} steps x3, spread {(max(got)-min(got))*1e3:.3f} ms)",
              flush=True)
        json.dump({"config": cfg, "rows": rows}, open(out_path, "w"), indent=1,
                  ensure_ascii=False)
    llm.shutdown()
    print("### done", flush=True)


if __name__ == "__main__":
    main()
