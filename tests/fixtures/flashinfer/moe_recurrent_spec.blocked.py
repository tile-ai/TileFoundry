"""Complete equation inventory for MoE, recurrent, state-space, and mHC.

Notes:
upstream: flashinfer-ai/flashinfer @ 2ab910c58fdd2392914ea05e2a8714946ac0eef6
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: selection/analysis
error: source defines no TileFoundry Module
classification: expected-spec notation; no authored Module is declared.
Equations retain unfused/fused boundaries and placement variants as test data.
"""

# noqa
M01_U = """def m01_unfused(x:gmem, bias:gmem):
  p = sigmoid(x); s = group_score(p, bias); g = topk(s, groups=4)
  e = topk(gather(p, g), k=8); return renorm(e) -> gmem
"""
M01 = (
    M01_U
    + """def m01_fused(x:gmem,bias:gmem):
  e = renorm(topk(group_topk(sigmoid(x)+bias, groups=4), k=8)) -> smem
  return e  # [program=token, cta=group, thread=expert, gpu=rank]
"""
)

M02_U = """def m02_unfused(x:gmem, w1:gmem):
  r = gather(x, route(x)) -> gmem; z = grouped_gemm(r,w1) -> gmem
  return silu_mul(z) -> gmem
"""
M02 = (
    M02_U
    + """def m02_fused(x:gmem,w1:gmem):
  z = silu_mul(grouped_gemm(gather(x,route(x)) -> smem,w1) -> smem) -> gmem
  return z  # [program=expert, cta=GEMM-MN, thread=fragment]
"""
)

M03_U = """def m03_unfused(z:gmem,w2:gmem,weights:gmem,route_ids:gmem):
  y = grouped_gemm(z,w2) -> gmem; return weighted_scatter_add(y,weights,route_ids)
"""
M03 = (
    M03_U
    + """def m03_fused(z:gmem,w2:gmem,weights:gmem,route_ids:gmem):
  y = weighted_scatter_add(grouped_gemm(z,w2) -> smem,weights,route_ids) -> gmem
  return y  # [program=expert, cta=output-tile, thread=lane, gpu=rank]
"""
)

M04_U = """def m04_unfused(x:gmem,w1:gmem,w2:gmem):
  r=route(x); a=grouped_gemm(gather(x,r),w1); h=silu_mul(a); y=grouped_gemm(h,w2)
  return scatter_add(y,r)
"""
M04 = (
    M04_U
    + """def m04_fused(x:gmem,w1:gmem,w2:gmem):
  return scatter_add(grouped_gemm(silu_mul(grouped_gemm(gather(x,route(x))->smem,w1)->smem),w2)->smem,route(x))
  # [program=token, cta=expert-tile, thread=mma-lane, gpu=rank]
"""
)

M05_U = """def m05_unfused(x:gmem,w1:gmem,w2:gmem):
  p=renorm(topk(score(x))); return reduce_experts([grouped_gemm(silu_mul(grouped_gemm(x,w1)),w2)],p)
"""
M05 = (
    M05_U
    + """def m05_fused(x:gmem,w1:gmem,w2:gmem):
  return reduce_experts_fused(x,w1,w2,route=renorm(topk(score(x)))) -> gmem
  # [program=token, cta=expert, thread=lane]
"""
)

M06_U = """def m06_unfused(x:gmem,rw:gmem,sw:gmem):
  y=routed_moe(x,rw); sh=shared_expert(x,sw); return y+sh
"""
M06 = (
    M06_U
    + """def m06_fused(x:gmem,rw:gmem,sw:gmem):
  return finalize(routed_moe(x,rw)->smem + shared_expert(x,sw)->smem) -> gmem
  # [program=token, cta=expert-tile, thread=accumulator, gpu=rank]
"""
)

M07 = """def m07_fused(x:gmem,w:gmem,scale:gmem):
  q=block_quant(x,scale)->smem; return finalize(grouped_gemm(q,w)->smem)->gmem
"""
M08 = """def m08_fused(x:gmem,w:gmem):
  q=pack_quant(x, dtype=fp4)->smem; return finalize(grouped_gemm(q,w)->smem)->gmem
"""
M09 = """def m09_fused(x:gmem,w:gmem):
  q=dequant(x,dtype=mxint4)->smem; return finalize(grouped_gemm(q,w)->smem)->gmem
"""
M10 = """def m10_fused(x:gmem,w:gmem,A:gmem,B:gmem):
  return grouped_gemm(x,w)->smem + bgmv(x,A,B)->smem -> gmem
"""
M11 = """def m11_fused(x:gmem,w:gmem,A:gmem,B:gmem,route:gmem):
  return finalize(grouped_gemm(x,w)->smem + bgmv(x,A,B,route)->smem)->gmem
"""
M12 = """def m12_fused(x:gmem,w:gmem,pack:gmem):
  r=route_pack(x,pack)->smem; return w4a16_expert_pipeline(r,w)->gmem
"""

# noqa
K01 = """def k01_fused(x:gmem,state:gmem,w:gmem):
  c=depthwise_conv4(x,w)->smem; u=silu(c); state2=kda_update(state,u)->gmem
  return rms_norm(state2)*silu(x)  # [program=token,cta=head,thread=channel,gpu=rank]
"""
K02 = """def k02_fused(q:gmem,k:gmem,v:gmem,state:gmem):
  qn=l2_norm(q)->smem; kn=l2_norm(k)->smem; b=sigmoid(beta(q,k)); s=kda_update(state,qn,kn,v,b)->gmem
  return s  # [program=sequence, cta=head, thread=channel]
"""
K03 = """def k03_fused(tokens:gmem,state:gmem,accepted:gmem):
  s=state
  for t in tokens: s=kda_step(s,t)->rmem
  return select_checkpoint(s,accepted)->gmem  # [program=sequence,cta=head,thread=channel]
"""

# noqa
S01 = """def s01_fused(x:gmem,A:gmem,B:gmem,C:gmem,D:gmem):
  z=chunk_cumsum(x,A)->smem; s=ssd_scan(z,B,C)->gmem
  return s + D*x  # [program=chunk,cta=sequence-tile,thread=channel,gpu=rank]
"""
S02 = """def s02_fused(old:gmem,new:gmem,checkpoint:gmem,pred:gmem):
  replay=ssu_replay(old,checkpoint)->smem; out,st=ssu_step(replay,new)->smem
  cache_write_if(pred,st)->gmem; return out  # [program=sequence,cta=chunk,thread=channel]
"""

# noqa
H01 = """def h01_fused(x:gmem,H:gmem):
  P=sinkhorn(H)->smem; y=residual_mix(x,P)->gmem
  return y  # [program=batch,cta=hidden-tile,thread=element,gpu=rank]
"""
H02 = """def h02_fused(x:gmem,H:gmem,gamma:gmem,beta:gmem):
  n=layer_norm(x,gamma,beta)->smem; P=sinkhorn(H)->smem
  return residual_mix(n,P)->gmem  # [program=batch,cta=hidden-tile,thread=element,gpu=rank]
"""

ALL = {
    k: v for k, v in globals().items() if k.startswith(("M", "K", "S", "H")) and k[1:2].isdigit()
}
