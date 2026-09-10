"""Complete equation inventory for FlashInfer norm/quant/RoPE boundaries.

Notes:
upstream: flashinfer-ai/flashinfer @ 2ab910c58fdd2392914ea05e2a8714946ac0eef6
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: selection/analysis
error: source defines no TileFoundry Module
classification: expected-spec notation; no authored Module is declared.
Notation retains storage handoffs, placement variants, and capability gaps.
"""

# noqa
N01 = """def rmsnorm_quant(x:gmem, gamma:gmem, scale:gmem):\n  y = rms_norm(x, gamma, eps=1e-6)\n  return cast(y / scale, fp8_e4m3fn)  # [program,cta-smem,thread-rmem]\n"""
N02 = """def add_rmsnorm(x:gmem, residual:gmem, gamma:gmem):\n  s = add(x, residual) -> smem\n  y = rms_norm(s, gamma, eps=1e-6)\n  return (s, y)  # [program,cta-smem,thread-rmem]\n"""
N03 = """def add_rmsnorm_quant(x:gmem, residual:gmem, gamma:gmem, scale:gmem):\n  s = add(x, residual) -> smem\n  y = rms_norm(s, gamma, eps=1e-6)\n  q = cast(y / scale, fp8_e4m3fn)\n  return (s, q)  # [program,cta-smem,thread-rmem]\n"""
N04 = """def gemma_add_rmsnorm(x:gmem, residual:gmem, gamma:gmem):\n  s = add(x, residual) -> smem\n  return s, rms_norm(s, add(gamma, 1), eps=1e-6)\n"""
N05 = """def layernorm_quant(x:gmem, gamma:gmem, beta:gmem, scale:gmem):\n  y = layer_norm(x, gamma, beta, axis=-1, eps=1e-5)\n  return cast(y / scale, fp8_e4m3fn)\n"""
N06 = """def rmsnorm_silu(x:gmem, gamma:gmem):\n  y = rms_norm(x, gamma, eps=1e-6) -> smem\n  return silu(y)\n"""
N07 = """def rmsnorm_silu_fp8(x:gmem, gamma:gmem):\n  y = rms_norm(x, gamma, eps=1e-6) -> smem\n  return cast(silu(y), fp8_e4m3fn)\n"""
N08 = """def rmsnorm_silu_nvfp4(x:gmem, gamma:gmem):\n  y = rms_norm(x, gamma, eps=1e-6) -> smem\n  return block_quant(silu(y), dtype=nvfp4, block=16)\n"""
N09 = """def qk_rmsnorm_rope(q:gmem, k:gmem, v:gmem, gq:gmem, gk:gmem, cos:gmem, sin:gmem):\n  qr = rope(rms_norm(q,gq), rms_norm(k,gk), cos, sin) -> smem\n  return qr.q, qr.k, v\n"""
N10 = N09.replace(
    "return qr.q, qr.k, v",
    "return quant(qr.q,fp8_e4m3fn), quant(qr.k,fp8_e4m3fn), quant(v,fp8_e4m3fn)",
)
N11 = """def dit_residual_ln(x:gmem, residual:gmem, scale:gmem, shift:gmem):\n  s = add(x,residual) -> smem\n  return layer_norm(s) * (1 + scale) + shift\n"""
N12 = """def dit_gate_residual_ln(x:gmem, residual:gmem, gate:gmem, scale:gmem, shift:gmem):\n  s = add(mul(x,gate),residual) -> smem\n  return layer_norm(s) * (1 + scale) + shift\n"""
N13 = """def dit_gate_residual_ln_gamma_beta(x:gmem, residual:gmem, gate:gmem, bias:gmem, gamma:gmem, beta:gmem):\n  s = add(mul(x,add(gate,bias)),residual) -> smem\n  return layer_norm(s,gamma,beta,axis=-1,eps=1e-5)\n"""

# noqa
Q01 = """def silu_mul_nvfp4(gate:gmem, up:gmem):\n  z = mul(silu(gate), up) -> smem\n  return block_quant(z, dtype=nvfp4, block=16)\n"""
Q02 = """def scaled_silu_mul_nvfp4(gate:gmem, up:gmem, expert_scale:gmem):\n  z = mul(silu(gate), up) * gather(expert_scale) -> smem\n  return block_quant(z, dtype=nvfp4, block=16)\n"""
Q03 = """def smooth_nvfp4(x:gmem, pre_scale:gmem):\n  return block_quant(x * pre_scale, dtype=nvfp4, block=16)\n"""
Q04 = """def mxfp4(x:gmem):\n  scale = block_absmax(x, block=32) -> smem\n  return pack_fp4(x / scale), scale\n"""
Q05 = """def mxfp8(x:gmem):\n  scale = block_absmax(x, block=32) -> smem\n  return cast(x / scale, fp8_e4m3fn), scale\n"""

# noqa
R01 = """def rope_quantize(q:gmem, k:gmem, cos:gmem, sin:gmem):\n  qr = rope(q,k,cos,sin) -> smem\n  return quant(qr.q,fp8_e4m3fn), quant(qr.k,fp8_e4m3fn)\n"""
R02 = """def mla_rope_quantize(q:gmem, k_rank2:gmem, cos:gmem, sin:gmem):\n  qr = rope(q,k_rank2,cos,sin) -> smem\n  return quant(qr.q,fp8_e4m3fn), quant(qr.k,fp8_e4m3fn)\n"""
R03 = """def rope_quant_append(q:gmem,k:gmem,cos:gmem,sin:gmem,cache:gmem,slots:gmem):\n  qr = rope(q,k,cos,sin) -> smem\n  q8,k8 = quant(qr,fp8_e4m3fn) -> rmem\n  cache[slots] = k8\n  return q8, cache\n"""
R04 = """def nvfp4_append(k:gmem,v:gmem,cache:gmem,slots:gmem):\n  kq = block_quant(k,dtype=nvfp4,block=16) -> rmem\n  vq = block_quant(v,dtype=nvfp4,block=16) -> rmem\n  cache[slots] = (kq,vq)\n  return cache\n"""
R05 = """def nvfp4_append_slot(k:gmem,v:gmem,cache:gmem,slot_map:gmem):\n  kq,vq = block_quant((k,v),dtype=nvfp4,block=16) -> rmem\n  cache[slot_map] = (kq,vq)\n  return cache\n"""

ALL = {
    **{f"N{i:02d}": globals()[f"N{i:02d}"] for i in range(1, 14)},
    **{f"Q{i:02d}": globals()[f"Q{i:02d}"] for i in range(1, 6)},
    **{f"R{i:02d}": globals()[f"R{i:02d}"] for i in range(1, 6)},
}
