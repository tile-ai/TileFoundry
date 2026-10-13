"""Count the FLOPs one prefill of `n` tokens costs, from the checkpoint itself.

Nothing here is typed in by hand: the shapes come out of the safetensors
headers, so the number moves when the checkpoint does. Two parts are added:

  matmul   2 * n * (every text weight that is a 2-D matmul)
           -- embedding is a lookup, not a matmul, so it is left out;
           -- norms, biases and the depthwise conv are 1-D, left out;
           -- the vision tower is left out (this bench feeds text only);
           -- lm_head is left out: sglang computes logits for the last
              position only, so it is 1 row, not n.

  scores   2 * n^2 * d_attn per full-attention layer, halved for causal
           -- q @ k^T and (scores) @ v, each 2*n^2*d_attn, causal keeps half,
              so the two together come to 2 * n^2 * d_attn.
           -- linear-attention layers are O(n) and are already paid for by
              their own projection weights in `matmul`.
"""
import json
import struct
from pathlib import Path


def _headers(ckpt: Path):
    for shard in sorted(ckpt.glob("*.safetensors")):
        with open(shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            yield shard.name, json.loads(f.read(hlen))


def weight_census(ckpt: Path) -> dict:
    """Split every tensor in the checkpoint into the buckets above."""
    out = {"matmul": 0, "embed": 0, "lm_head": 0, "vision": 0, "vector": 0}
    for _, hdr in _headers(ckpt):
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            shape = meta["shape"]
            numel = 1
            for d in shape:
                numel *= d
            if "visual" in name or "vision" in name:
                bucket = "vision"
            elif "embed_tokens" in name:
                bucket = "embed"
            elif name.startswith("lm_head") or ".lm_head" in name:
                bucket = "lm_head"
            elif len(shape) < 2:
                bucket = "vector"
            else:
                bucket = "matmul"
            out[bucket] += numel
    return out


def attn_dims(ckpt: Path) -> tuple[int, int]:
    """(number of full-attention layers, q width) from config.json."""
    cfg = json.loads((ckpt / "config.json").read_text())
    txt = cfg.get("text_config", cfg)
    types = txt.get("layer_types")
    if types is None:
        n_full = txt["num_hidden_layers"]
    else:
        n_full = sum(1 for t in types if t == "full_attention")
    d_attn = txt["num_attention_heads"] * txt["head_dim"]
    return n_full, d_attn


def prefill_flops(ckpt: Path, n: int) -> dict:
    c = weight_census(ckpt)
    n_full, d_attn = attn_dims(ckpt)
    matmul = 2 * n * c["matmul"]
    scores = 2 * n * n * d_attn * n_full
    return {"matmul": matmul, "scores": scores, "total": matmul + scores,
            "census": c, "n_full_attention": n_full, "d_attn": d_attn}


if __name__ == "__main__":
    import sys
    ck = Path(sys.argv[1])
    c = weight_census(ck)
    n_full, d_attn = attn_dims(ck)
    print(f"ckpt            {ck}")
    for k, v in c.items():
        print(f"  {k:<10}{v/1e9:9.3f} B params")
    print(f"  full-attn layers {n_full}, d_attn {d_attn}")
    for n in (int(x) for x in (sys.argv[2:] or ["1024", "4096", "8192"])):
        f = prefill_flops(ck, n)
        print(f"n={n:>6}  matmul {f['matmul']/1e12:8.1f} T  "
              f"scores {f['scores']/1e12:7.1f} T  "
              f"total {f['total']/1e12:8.1f} TFLOP  "
              f"({100*f['scores']/f['total']:4.1f}% scores)")
