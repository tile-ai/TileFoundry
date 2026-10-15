#!/usr/bin/env python3
"""The transformers reference for every prompt length.

For each T in (128, 256, 512, 1024) this runs the published model class
(`Qwen3_5ForConditionalGeneration`, bf16, text only, `use_cache=True`) on the
first T tokens of prompts/prompt.txt and writes, under reference/T{T}/:

  token_ids.pt  cos.pt  sin.pt    the three activations `model.py:prefill` takes
  expected.pt                     [logits(1,V), K(16,T,4,256), V(16,T,4,256),
                                   conv window(48,10240,3), recurrent state(48,48,128,128) f32]
  meta.json                       first token, attention implementation, versions

The conv window is the last kernel-1 columns of the cache's conv state (the
column HF keeps in front is the one the next decode step drops).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import paths  # noqa: E402
import prompting  # noqa: E402


def main() -> None:
    device = "cuda:0"
    config = json.loads((ROOT / "config.json").read_text())["text_config"]
    ckpt = str(paths.need("ckpt"))
    tok = AutoTokenizer.from_pretrained(ckpt)
    text = (ROOT / "prompts/prompt.txt").read_text()
    t0 = time.time()
    hf = Qwen3_5ForConditionalGeneration.from_pretrained(ckpt, dtype=torch.bfloat16).to(device)
    hf.eval()
    print(f"loaded in {time.time() - t0:.0f} s; attn_implementation={hf.config._attn_implementation}", flush=True)
    kinds = config["layer_types"]
    lengths = [int(a) for a in sys.argv[1:]] or list(prompting.LENGTHS)
    for T in lengths:
        ids = prompting.prompt_ids(tok, text, T)
        input_ids = torch.tensor([ids], dtype=torch.int64, device=device)
        with torch.no_grad():
            out = hf(input_ids=input_ids, use_cache=True)
        logits = out.logits[:, -1, :].contiguous()
        cache = out.past_key_values
        ks, vs, convs, recs = [], [], [], []
        for i, kind in enumerate(kinds):
            layer = cache.layers[i]
            if kind == "full_attention":
                ks.append(layer.keys[0].transpose(0, 1).contiguous())  # (T, 4, 256)
                vs.append(layer.values[0].transpose(0, 1).contiguous())
            else:
                conv = layer.conv_states
                rec = layer.recurrent_states
                conv = conv[0] if isinstance(conv, (list, tuple, dict)) else conv
                rec = rec[0] if isinstance(rec, (list, tuple, dict)) else rec
                convs.append(conv.reshape(-1, conv.shape[-2], conv.shape[-1])[0, :, 1:].contiguous())
                recs.append(rec.reshape(-1, *rec.shape[-3:])[0].contiguous())
        expected = [logits, torch.stack(ks), torch.stack(vs), torch.stack(convs), torch.stack(recs)]
        # cos/sin exactly as the model's own rotary module builds them for these positions.
        rotary = hf.model.language_model.rotary_emb
        pos = torch.arange(T, device=device).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = rotary(out.logits.new_zeros(1, dtype=torch.bfloat16), pos)
        cos, sin = cos[0].contiguous(), sin[0].contiguous()
        mine_cos, mine_sin = prompting.rope_tables(T, config, device)
        d = ROOT / "reference" / f"T{T}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(input_ids[0].cpu(), d / "token_ids.pt")
        torch.save(cos.cpu(), d / "cos.pt")
        torch.save(sin.cpu(), d / "sin.pt")
        torch.save([t.cpu() for t in expected], d / "expected.pt")
        top2 = torch.topk(logits.float(), 2, dim=-1)
        meta = dict(
            T=T,
            first_token=int(top2.indices[0, 0]),
            first_token_text=tok.decode([int(top2.indices[0, 0])]),
            top2_margin=float(top2.values[0, 0] - top2.values[0, 1]),
            attn_implementation=hf.config._attn_implementation,
            rope_tables_equal_to_prompting=bool(torch.equal(cos, mine_cos) and torch.equal(sin, mine_sin)),
            shapes=[list(t.shape) for t in expected],
            dtypes=[str(t.dtype) for t in expected],
            torch=torch.__version__,
            transformers=transformers.__version__,
        )
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
