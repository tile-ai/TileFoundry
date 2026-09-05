"""Run one decode step twice, on real weights, and compare the two implementations.

`check_all.py` compares a twin against the authored semantics and needs the whole
model evaluated to do it. This is the cheaper gate that comes first: load once,
run two `NEMO_IMPL` settings over the same inputs and states, and report the
logits and every state tensor.

    python compare_impls.py --a ops --b cuda [--steps 3]
"""
from __future__ import annotations

import argparse

import torch

import paths


def _owners():
    """One label per `fresh` tensor, in the order a step returns them."""
    import model  # noqa: PLC0415

    out = []
    for i, kind in enumerate(model.LAYER_KINDS):
        if kind == "linear_attention":
            out += [f"L{i} mamba conv_out", f"L{i} mamba ssm_out"]
        elif kind == "full_attention":
            out += [f"L{i} attn k_new", f"L{i} attn v_new"]
    return out


_OWNER = None


def _owner(n):
    global _OWNER
    if _OWNER is None:
        _OWNER = _owners()
    return _OWNER[n] if n < len(_OWNER) else f"#{n}"


def rel_l2(a, b):
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    d = b.norm().item()
    return (a - b).norm().item() / (d if d else 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--a", default="ops")
    ap.add_argument("--b", default="cuda")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--max-rel", type=float, default=None,
                    help="default: the bound check_all.py derives for the logits")
    a = ap.parse_args()

    # No invented tolerance. The bound is the one `check_all.py` argues from
    # how many times a value lands in bf16 on the way to the logits: one
    # round-to-nearest is at most 2^-9 relative, and k of them sum as a random
    # walk. Two implementations that round differently at every layer are
    # allowed to differ by that much and no more.
    if a.max_rel is None:
        import math  # noqa: PLC0415

        import check_all  # noqa: PLC0415
        import model as _m  # noqa: PLC0415
        k = sum(check_all._LANDINGS[kind] for kind in _m.LAYER_KINDS) + 3
        a.max_rel = 2.0 ** -9 * math.sqrt(k)
        print(f"bound 2^-9*sqrt({k}) = {a.max_rel:.3e}  (check_all.py's derivation)")

    from tilefoundry.runtime import SafetensorsResource  # noqa: PLC0415
    import runtime_model  # noqa: PLC0415

    twin = runtime_model.Nemotron35Lightning30BA3BRuntime()
    twin.load(SafetensorsResource(str(paths.need("prepared", a.prepared)), device=a.device))

    def run(impl):
        runtime_model.set_impl(impl)
        caches = twin.init_caches(device=a.device, capacity=a.steps + 4)
        ids = torch.full((a.steps + 4,), 1784, dtype=torch.int64, device=a.device)
        outs = []
        for step in range(a.steps):
            args = twin.prepare_inputs_for_generation(ids[: step + 1], step, caches,
                                                      device=a.device)
            logits, fresh = twin.forward(*args)
            caches = twin.append_cache(caches, fresh)
            outs.append((logits.detach().clone(),
                         [f.detach().clone() for f in fresh]))
        return outs

    ra, rb = run(a.a), run(a.b)
    worst, bad = 0.0, 0
    for step, ((la, fa), (lb, fb)) in enumerate(zip(ra, rb)):
        r = rel_l2(lb, la)
        worst = max(worst, r)
        top_a = int(torch.argmax(la.reshape(-1)).item())
        top_b = int(torch.argmax(lb.reshape(-1)).item())
        same = "same" if top_a == top_b else f"DIFFER {top_a} vs {top_b}"
        bad += top_a != top_b
        print(f"step {step}  logits rel_l2 {r:.3e}   argmax {same}")
        # Name which tensor is worst, and which layer produced it: a number
        # with no owner says a step is wrong without saying where.
        rows = [(rel_l2(y, x), n) for n, (x, y) in enumerate(zip(fa, fb))]
        rows.sort(reverse=True)
        fr = rows[0][0] if rows else 0.0
        worst = max(worst, fr)
        print(f"          worst state rel_l2 {fr:.3e}  over {len(fa)} tensors")
        for r, n in rows[:3]:
            print(f"            [{n:2d}] {_owner(n):28} {r:.3e}")
    over = worst > a.max_rel
    print(f"\nworst rel_l2 {worst:.3e}  (limit {a.max_rel:.1e})   "
          f"{'FAIL' if over or bad else 'PASS'}")
    return 1 if (over or bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
