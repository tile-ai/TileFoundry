"""Qwen3.5-35B-A3B decode-step fusion functions (main-agent authored).

主 agent 划界产物：每个 ``@func`` 是一个 fusion 边界 = 未来一个 RuntimeModule
方法 = sub-agent 一个 kernel 的语义契约。语义逐行对照
``transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py``（下称 M，
transformers 5.12.1）。

结构（config.json 已核）：40 层 = 30×linear_attention（GatedDeltaNet）+
10×full_attention（layer_types 为准）；每层 MoE（256 专家 top-8 + 标量门
shared expert）。hidden=2048，vocab=248320。

关键语义备忘（源码为准，实现/加载必须遵守）：
- RMSNorm（M:817）：权重以 ``(1 + w)`` 参与——checkpoint 存 w-1 形式。三个
  mix 函数（moe_mix/full_attn_mix/gdn_mix）现在统一收 **RAW**（未 +1）gamma，
  ``(1 + w)`` 挪进函数体就地算；对应的 ``*_convert`` @func（moe_convert/
  attn_convert/gdn_convert）是唯一的 RAW ckpt → CANONICAL（= kernel 原生
  layout）转换入口，纯 repack/cast，无数值变换。
- full-attn（M:643）：q_proj 输出 [query|gate]（每头 512 维 = 256 query +
  256 gate，M:684 view+chunk）；per-head q/k RMSNorm（256 维，1+w）；partial
  rope 仅前 64 维，rotate-half 约定（M:560，与 tf.rope 相同），theta=1e7；
  文本下 mrope 交织退化为恒等（三组 position 相同，M:176 覆写 no-op）；
  attn_out × sigmoid(gate) 逐元素（M:717）后 o_proj。scaling=256^-0.5。
- MoE（M:776/795）：router logits→softmax(f32 全 256)→top8→8 权重归一→
  cast 回 bf16；experts 堆叠 gate_up[256,1024,2048] / down[256,2048,512]，
  silu(gate)*up→down；shared expert（inter 512）乘 **标量** sigmoid 门
  （Linear 2048→1，M:807/813）。
- GatedDeltaNet（M:369）单步：in_proj_qkv(2048→8192) → 分组 conv1d(k=4,
  silu, bias=False) 状态推进（M:220）→ split q(2048)/k(2048)/v(4096)，
  q/k 16 头 repeat_interleave→32（M:518）；beta=sigmoid(in_proj_b(x))[32]；
  g = -exp(A_log)·softplus(in_proj_a(x)+dt_bias)[32]（f32，M:516）；递推
  （f32，q/k 先 l2norm，scale=128^-0.5，M:325-368）：S←S·e^g；
  kv_mem=Σ_k S·k；delta=(v-kv_mem)·β；S←S+k⊗delta；out=Σ_k S·q；
  RMSNormGated：norm(out)·(1+w)·silu(z)（M:185，z=in_proj_z(x)）→ out_proj。
  GDN 的 HIR 体待补（先钉签名与语义，oracle 用 HF 层实例）。

fusion 划界（每函数 = 一个 kernel 交付单元）：
  full_attn_mix : in_norm + qkv/gate 投影 + per-head norm + partial rope +
                  cache 写 + attend + sigmoid 门 + o_proj + residual
  moe_mix       : post_norm + router + top8 专家 + 标量门 shared + residual
  （gdn_mix 签名见文件底部注释）

attn 的因果掩码沿用 qwen3_5_30b_a3b fixture 的约定：编排层提供加性 mask
（有效前缀 0 / 其余 -inf），HIR 侧 ``scores + mask``。
"""
from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.core.kinds import ReduceKind

HIDDEN = 2048
N_Q_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 256
ROT_DIM = 64          # partial_rotary_factor 0.25 × 256
Q_PROJ = N_Q_HEADS * HEAD_DIM              # 4096
QG_PROJ = Q_PROJ * 2                       # 8192: 每头 [query(256) | gate(256)]
KV_PROJ = N_KV_HEADS * HEAD_DIM            # 512
GQA_GROUP = N_Q_HEADS // N_KV_HEADS        # 8
SCALE = HEAD_DIM ** -0.5

N_EXPERTS = 256
TOP_K = 8
MOE_INTER = 512
PACKED_EXPERTS = N_EXPERTS + 1  # 256 routed + shared packed at slot N_EXPERTS（镜像 kernels/moe.py）
VOCAB = 248320

CACHE_CAP = 4096


@func
def moe_mix(
    x: Tensor[(1, 1, HIDDEN), "bf16"],                       # 层内残差流（token mixer 之后）
    post_norm_gamma_raw: ConstTensor[(HIDDEN,), "f32"],      # RAW（未 +1）；(1+w) 挪进函数体
    router_w: ConstTensor[(N_EXPERTS, HIDDEN), "bf16"],
    packed_gate_up: ConstTensor[(PACKED_EXPERTS, 2 * MOE_INTER, HIDDEN), "bf16"],
    packed_down: ConstTensor[(PACKED_EXPERTS, HIDDEN, MOE_INTER), "bf16"],
    shared_gate_w: ConstTensor[(1, HIDDEN), "bf16"],
) -> Tensor[(1, 1, HIDDEN), "bf16"]:
    # CANONICAL 权重形式 = kernel 的 packed layout（唯一来源，见 moe_convert）：
    # shared expert 打包为 packed_gate_up/packed_down 的第 N_EXPERTS 槽（镜像
    # kernels/moe.py 的 PACKED_EXPERTS）。evaluator（本函数）和 kernel override
    # 读同一份 self.weights——不再有第二套 func-canonical 权重需要对齐。
    # ── post_attention_layernorm（M:871）：RAW 权重，(1+w) 就地算 ────────
    h = tf.rms_norm(x, 1.0 + post_norm_gamma_raw)
    ht = tf.reshape(h, new_shape=(1, HIDDEN))

    # ── router（M:776）: softmax(f32 全体) → top8 → 归一 ─────────────
    logits = tf.matmul(
        tf.cast(ht, dtype="f32"),
        tf.transpose(tf.cast(router_w, dtype="f32"), perm=(1, 0)),
    )                                                        # (1, 256) f32
    lmax = tf.reduce(logits, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
    e = tf.exp(logits - lmax)
    probs = e / tf.reduce(e, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    top_p, eids = tf.topk(probs, k=TOP_K, axis=-1)           # (1,8) f32 / i64
    gweights = top_p / tf.reduce(top_p, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    gweights = tf.cast(gweights, dtype="bf16")

    # ── 8 个被选专家（M:737）: silu(gate)*up → down，加权和 ──────────
    # packed_* 是 [257,...]；eids ∈ [0,256) 只取路由槽，语义与旧 [256,...] 一致。
    g_up = tf.gather(packed_gate_up, eids, axis=0)           # (1,8,1024,2048) bf16
    down = tf.gather(packed_down, eids, axis=0)              # (1,8,2048,512)  bf16
    tok = tf.reshape(tf.cast(ht, dtype="bf16"), new_shape=(1, 1, HIDDEN, 1))
    fused = tf.matmul(g_up, tok)                             # (1,8,1024,1)
    gate = tf.slice(fused, begin=(0, 0, 0, 0), end=(1, TOP_K, MOE_INTER, 1), strides=(1, 1, 1, 1))
    up = tf.slice(fused, begin=(0, 0, MOE_INTER, 0), end=(1, TOP_K, 2 * MOE_INTER, 1), strides=(1, 1, 1, 1))
    act = gate * tf.sigmoid(gate) * up                       # silu(gate)·up, (1,8,512,1)
    per_expert = tf.reshape(tf.matmul(down, act), new_shape=(1, TOP_K, HIDDEN))
    weighted = per_expert * tf.reshape(gweights, new_shape=(1, TOP_K, 1))
    routed = tf.reduce(weighted, axes=(1,), keepdim=False, kind=ReduceKind.SUM)  # (1, 2048)

    # ── shared expert × 标量 sigmoid 门（M:807-813）──────────────────
    # packed 槽 N_EXPERTS：gate_up 前 512 行=gate，后 512 行=up（镜像
    # moe_convert 里 concat(shared_gate, shared_up, axis=0) 的顺序）。
    s_gu = tf.slice(
        packed_gate_up, begin=(N_EXPERTS, 0, 0),
        end=(PACKED_EXPERTS, 2 * MOE_INTER, HIDDEN), strides=(1, 1, 1),
    )
    s_gu = tf.reshape(s_gu, new_shape=(2 * MOE_INTER, HIDDEN))
    shared_gate = tf.slice(s_gu, begin=(0, 0), end=(MOE_INTER, HIDDEN), strides=(1, 1))
    shared_up = tf.slice(s_gu, begin=(MOE_INTER, 0), end=(2 * MOE_INTER, HIDDEN), strides=(1, 1))
    s_down = tf.slice(
        packed_down, begin=(N_EXPERTS, 0, 0),
        end=(PACKED_EXPERTS, HIDDEN, MOE_INTER), strides=(1, 1, 1),
    )
    shared_down = tf.reshape(s_down, new_shape=(HIDDEN, MOE_INTER))

    hb = tf.cast(ht, dtype="bf16")
    sg = tf.matmul(hb, tf.transpose(shared_gate, perm=(1, 0)))
    su = tf.matmul(hb, tf.transpose(shared_up, perm=(1, 0)))
    s_act = sg * tf.sigmoid(sg) * su                          # (1, 512)
    s_out = tf.matmul(s_act, tf.transpose(shared_down, perm=(1, 0)))          # (1, 2048)
    gate_scalar = tf.sigmoid(tf.matmul(hb, tf.transpose(shared_gate_w, perm=(1, 0))))  # (1,1)
    s_gated = s_out * gate_scalar

    # ── 残差（M:874）─────────────────────────────────────────────────
    y = tf.reshape(routed + s_gated, new_shape=(1, 1, HIDDEN))
    return x + y


@func
def moe_convert(
    post_norm: ConstTensor[(HIDDEN,), "f32"],                          # RAW ckpt layernorm 权重（未 +1）
    gate_w: ConstTensor[(N_EXPERTS, HIDDEN), "bf16"],
    experts_gate_up: ConstTensor[(N_EXPERTS, 2 * MOE_INTER, HIDDEN), "bf16"],
    experts_down: ConstTensor[(N_EXPERTS, HIDDEN, MOE_INTER), "bf16"],
    shared_gate: ConstTensor[(MOE_INTER, HIDDEN), "bf16"],
    shared_up: ConstTensor[(MOE_INTER, HIDDEN), "bf16"],
    shared_down: ConstTensor[(HIDDEN, MOE_INTER), "bf16"],
    shared_gate_w: ConstTensor[(1, HIDDEN), "bf16"],
):
    # 返回 (post_norm_raw, router_w, packed_gate_up, packed_down, shared_gate_w)——
    # moe_mix 的 CANONICAL（= kernel packed layout）权重。唯一 convert 入口：
    # evaluator（经 moe_mix）和 kernel override 都读这一份产物，不再各自转换、
    # 不再需要对齐两份权重。shared expert 打包进 packed_* 的第 N_EXPERTS 槽，
    # gate_up 顺序 concat(shared_gate, shared_up)（镜像 kernels/moe.py 的
    # convert_hf_weights，纯 repack，无数值变换）。
    shared_gate_up = tf.concat(shared_gate, shared_up, axis=0)             # (1024, 2048)
    shared_gate_up = tf.reshape(shared_gate_up, new_shape=(1, 2 * MOE_INTER, HIDDEN))
    packed_gate_up = tf.concat(experts_gate_up, shared_gate_up, axis=0)    # (257, 1024, 2048)

    shared_down_e = tf.reshape(shared_down, new_shape=(1, HIDDEN, MOE_INTER))
    packed_down = tf.concat(experts_down, shared_down_e, axis=0)          # (257, 2048, 512)

    post_norm_raw = tf.cast(post_norm, dtype="f32")
    router_w = tf.cast(gate_w, dtype="bf16")
    return post_norm_raw, router_w, packed_gate_up, packed_down, tf.cast(shared_gate_w, dtype="bf16")


@func
def full_attn_mix(
    x: Tensor[(1, 1, HIDDEN), "bf16"],                       # 层输入残差流
    in_norm_raw: ConstTensor[(HIDDEN,), "f32"],              # RAW（未 +1）；(1+w) 挪进函数体
    w_qg: ConstTensor[(HIDDEN, QG_PROJ), "bf16"],
    w_k: ConstTensor[(HIDDEN, KV_PROJ), "bf16"],
    w_v: ConstTensor[(HIDDEN, KV_PROJ), "bf16"],
    w_o: ConstTensor[(Q_PROJ, HIDDEN), "bf16"],
    q_norm_raw: ConstTensor[(HEAD_DIM,), "f32"],             # RAW（未 +1）
    k_norm_raw: ConstTensor[(HEAD_DIM,), "f32"],
    cos_cache: Tensor[(CACHE_CAP, ROT_DIM), "f32"],          # rope 表（全长，pos_ids 索引）
    sin_cache: Tensor[(CACHE_CAP, ROT_DIM), "f32"],
    pos_ids: Tensor[(1, 1), "i32"],                          # 当前位置 id
    k_cache: Tensor[(1, CACHE_CAP, N_KV_HEADS, HEAD_DIM), "bf16"],
    v_cache: Tensor[(1, CACHE_CAP, N_KV_HEADS, HEAD_DIM), "bf16"],
    pos: Tensor[(1,), "i32"],                                # 写入槽位 = 已有长度
    s_one: Tensor[(1,), "i32"],                              # 常量 1（cache_update 的 s）
    attn_mask: Tensor[(1, 1, 1, CACHE_CAP), "f32"],          # 加性：[0,pos] 为 0，其余 -inf
):
    # 返回 (y[1,1,2048], k_cache', v_cache')——多输出按仓库惯例不写返回注解（推断）。
    # CANONICAL 权重形式 = kernels/full_attn.py KernelWeights 的原生 layout
    # （唯一来源，见 attn_convert）：q/k/v/o 已转置 [in,out]，3 个 norm 保持
    # RAW——evaluator（本函数）和 kernel override 读同一份 self.weights。
    # ── input_layernorm（M:855，RAW 权重，(1+w) 就地算）───────────────
    h = tf.rms_norm(x, 1.0 + in_norm_raw)

    # ── q_proj → 每头 [query|gate]（M:684-688）───────────────────────
    qg = tf.reshape(tf.matmul(h, w_qg), new_shape=(1, 1, N_Q_HEADS, 2 * HEAD_DIM))
    q = tf.slice(qg, begin=(0, 0, 0, 0), end=(1, 1, N_Q_HEADS, HEAD_DIM), strides=(1, 1, 1, 1))
    gate = tf.slice(qg, begin=(0, 0, 0, HEAD_DIM), end=(1, 1, N_Q_HEADS, 2 * HEAD_DIM), strides=(1, 1, 1, 1))
    k = tf.reshape(tf.matmul(h, w_k), new_shape=(1, 1, N_KV_HEADS, HEAD_DIM))
    v = tf.reshape(tf.matmul(h, w_v), new_shape=(1, 1, N_KV_HEADS, HEAD_DIM))

    # ── per-head RMSNorm（M:690-692，RAW 权重，(1+w) 就地算）──────────
    q = tf.rms_norm(q, 1.0 + q_norm_raw)
    k = tf.rms_norm(k, 1.0 + k_norm_raw)

    # ── partial rope：前 64 维 rotate-half（M:568）───────────────────
    q_rot = tf.slice(q, begin=(0, 0, 0, 0), end=(1, 1, N_Q_HEADS, ROT_DIM), strides=(1, 1, 1, 1))
    q_pass = tf.slice(q, begin=(0, 0, 0, ROT_DIM), end=(1, 1, N_Q_HEADS, HEAD_DIM), strides=(1, 1, 1, 1))
    k_rot = tf.slice(k, begin=(0, 0, 0, 0), end=(1, 1, N_KV_HEADS, ROT_DIM), strides=(1, 1, 1, 1))
    k_pass = tf.slice(k, begin=(0, 0, 0, ROT_DIM), end=(1, 1, N_KV_HEADS, HEAD_DIM), strides=(1, 1, 1, 1))
    q_rot_r, k_rot_r = tf.rope(
        tf.cast(q_rot, dtype="f32"), tf.cast(k_rot, dtype="f32"), cos_cache, sin_cache, pos_ids
    )
    q = tf.concat(tf.cast(q_rot_r, dtype="bf16"), q_pass, axis=-1)
    k = tf.concat(tf.cast(k_rot_r, dtype="bf16"), k_pass, axis=-1)

    # ── cache 写入当前位置（cache_update: new[:, :s] → cache[:, pos:pos+s]）─
    k_cache_new = tf.cache_update(k_cache, pos, s_one, k)
    v_cache_new = tf.cache_update(v_cache, pos, s_one, v)

    # ── attend：GQA 8:1，加性掩码（M:618 eager，scaling 融进 q）──────
    qh = tf.reshape(q, new_shape=(1, N_Q_HEADS, 1, HEAD_DIM))
    qh = tf.cast(qh, dtype="f32") * tf.full_like(tf.cast(qh, dtype="f32"), value=SCALE)
    kh = tf.transpose(k_cache_new, perm=(0, 2, 1, 3))         # (1, 2, CAP, 256)
    vh = tf.transpose(v_cache_new, perm=(0, 2, 1, 3))
    kh = tf.repeat_interleave(kh, repeats=GQA_GROUP, axis=1)  # (1, 16, CAP, 256)
    vh = tf.repeat_interleave(vh, repeats=GQA_GROUP, axis=1)
    scores = tf.matmul(qh, tf.transpose(tf.cast(kh, dtype="f32"), perm=(0, 1, 3, 2)))  # (1,16,1,CAP) f32
    scores = scores + attn_mask
    smax = tf.reduce(scores, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
    p = tf.exp(scores - smax)
    p = p / tf.reduce(p, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    o = tf.matmul(tf.cast(p, dtype="bf16"), vh)               # (1,16,1,256)

    # ── sigmoid 输出门（M:716-717）→ o_proj → 残差 ───────────────────
    o = o * tf.sigmoid(tf.reshape(gate, new_shape=(1, N_Q_HEADS, 1, HEAD_DIM)))
    o_flat = tf.reshape(tf.transpose(o, perm=(0, 2, 1, 3)), new_shape=(1, 1, Q_PROJ))
    y = tf.matmul(o_flat, w_o)
    return x + y, k_cache_new, v_cache_new


@func
def attn_convert(
    input_layernorm: ConstTensor[(HIDDEN,), "f32"],          # RAW ckpt layernorm 权重（未 +1）
    q_proj: ConstTensor[(QG_PROJ, HIDDEN), "bf16"],           # nn.Linear 原生 [out,in]
    k_proj: ConstTensor[(KV_PROJ, HIDDEN), "bf16"],
    v_proj: ConstTensor[(KV_PROJ, HIDDEN), "bf16"],
    o_proj: ConstTensor[(HIDDEN, Q_PROJ), "bf16"],            # nn.Linear 原生 [out,in]
    q_norm: ConstTensor[(HEAD_DIM,), "f32"],                  # RAW ckpt q_norm 权重（未 +1）
    k_norm: ConstTensor[(HEAD_DIM,), "f32"],
):
    # 返回 (in_norm_raw, w_qg, w_k, w_v, w_o, q_norm_raw, k_norm_raw)——
    # full_attn_mix 的 CANONICAL（= kernels/full_attn.py KernelWeights 原生
    # layout）权重。唯一 convert 入口：evaluator（经 full_attn_mix）和 kernel
    # override 都读这一份产物。q/k/v/o 转置成 [in,out]（镜像
    # fa.convert_hf_weights 的 .t()），norm 保持 RAW，(1+w) 挪进
    # full_attn_mix 函数体；纯 repack/cast，无数值变换。
    in_norm_raw = tf.cast(input_layernorm, dtype="f32")
    w_qg = tf.cast(tf.transpose(q_proj, perm=(1, 0)), dtype="bf16")
    w_k = tf.cast(tf.transpose(k_proj, perm=(1, 0)), dtype="bf16")
    w_v = tf.cast(tf.transpose(v_proj, perm=(1, 0)), dtype="bf16")
    w_o = tf.cast(tf.transpose(o_proj, perm=(1, 0)), dtype="bf16")
    q_norm_raw = tf.cast(q_norm, dtype="f32")
    k_norm_raw = tf.cast(k_norm, dtype="f32")
    return in_norm_raw, w_qg, w_k, w_v, w_o, q_norm_raw, k_norm_raw


GDN_KEY_DIM = 2048       # 16 头 × 128
GDN_VALUE_DIM = 4096     # 32 头 × 128
GDN_CONV_DIM = 8192      # key×2 + value
GDN_HEADS = 32
GDN_HDIM = 128
GDN_CONV_K = 4
GDN_EPS = 1e-6           # l2norm（M:238）与 RMSNormGated 同 eps
GDN_SCALE = GDN_HDIM ** -0.5
GDN_INV_HDIM = 1.0 / GDN_HDIM


@func
def gdn_mix(
    x: Tensor[(1, 1, HIDDEN), "bf16"],
    in_norm_raw: ConstTensor[(HIDDEN,), "f32"],              # RAW（未 +1）；(1+w) 挪进函数体
    w_qkv: ConstTensor[(GDN_CONV_DIM, HIDDEN), "bf16"],      # native [out,in]（kernel 原生，见 gdn_convert）
    w_z: ConstTensor[(GDN_VALUE_DIM, HIDDEN), "bf16"],
    w_b: ConstTensor[(GDN_HEADS, HIDDEN), "bf16"],
    w_a: ConstTensor[(GDN_HEADS, HIDDEN), "bf16"],
    conv_w: ConstTensor[(GDN_CONV_DIM, GDN_CONV_K), "bf16"],
    a_log: ConstTensor[(GDN_HEADS,), "f32"],
    dt_bias: ConstTensor[(GDN_HEADS,), "f32"],
    gated_norm_gamma: ConstTensor[(GDN_HDIM,), "bf16"],      # 原样权重！RMSNormGated（M:187）ones 初始化直接乘，无 (1+w)
    w_out: ConstTensor[(HIDDEN, GDN_VALUE_DIM), "bf16"],     # native [out,in]（kernel 原生）
    conv_state: Tensor[(1, GDN_CONV_DIM, GDN_CONV_K), "bf16"],
    rec_state: Tensor[(1, GDN_HEADS, GDN_HDIM, GDN_HDIM), "f32"],
):
    # 返回 (y[1,1,2048], conv_state', rec_state')。语义 M:369-558 decode 分支。
    # CANONICAL 权重形式 = kernels/linear_attn.py LinearAttnWeights 的原生
    # layout（唯一来源，见 gdn_convert）：w_qkv/w_z/w_b/w_a/w_out 是 nn.Linear
    # 原生 [out,in]，本函数体内转置后 matmul——evaluator（本函数）和 kernel
    # override 读同一份 self.weights。
    # ── input_layernorm（RAW 权重，(1+w) 就地算）──────────────────────
    h = tf.rms_norm(x, 1.0 + in_norm_raw)

    # ── in_proj_qkv → conv1d 单步（M:220-236：拼状态、留尾 4、depthwise
    #    卷积 = 沿核维加权和、silu）─────────────────────────────────────
    mixed = tf.reshape(tf.matmul(h, tf.transpose(w_qkv, perm=(1, 0))), new_shape=(1, GDN_CONV_DIM, 1))
    tail = tf.slice(
        conv_state, begin=(0, 0, 1), end=(1, GDN_CONV_DIM, GDN_CONV_K), strides=(1, 1, 1)
    )
    conv_state_new = tf.concat(tail, mixed, axis=-1)          # (1,8192,4)
    conv_out = tf.reduce(
        conv_state_new * tf.reshape(conv_w, new_shape=(1, GDN_CONV_DIM, GDN_CONV_K)),
        axes=(-1,), keepdim=False, kind=ReduceKind.SUM,
    )                                                         # (1,8192)
    conv_out = conv_out * tf.sigmoid(conv_out)                # silu（M:233）

    # ── split q/k/v，16 头 → repeat_interleave → 32（M:500/518）──────
    q = tf.reshape(
        tf.slice(conv_out, begin=(0, 0), end=(1, GDN_KEY_DIM), strides=(1, 1)),
        new_shape=(1, GDN_HEADS // 2, GDN_HDIM),
    )
    k = tf.reshape(
        tf.slice(conv_out, begin=(0, GDN_KEY_DIM), end=(1, 2 * GDN_KEY_DIM), strides=(1, 1)),
        new_shape=(1, GDN_HEADS // 2, GDN_HDIM),
    )
    v = tf.reshape(
        tf.slice(conv_out, begin=(0, 2 * GDN_KEY_DIM), end=(1, GDN_CONV_DIM), strides=(1, 1)),
        new_shape=(1, GDN_HEADS, GDN_HDIM),
    )
    q = tf.repeat_interleave(q, repeats=2, axis=1)            # (1,32,128)
    k = tf.repeat_interleave(k, repeats=2, axis=1)

    # ── 门（M:514-516，f32）: β=sigmoid(b)，g=-exp(A_log)·softplus(a+dt_bias)
    # β：sigmoid 在 bf16（M:513，投影 dtype）之后才进 f32 递推
    beta = tf.cast(
        tf.sigmoid(tf.reshape(tf.matmul(h, tf.transpose(w_b, perm=(1, 0))), new_shape=(1, GDN_HEADS))),
        dtype="f32",
    )
    a_prj = tf.cast(
        tf.reshape(tf.matmul(h, tf.transpose(w_a, perm=(1, 0))), new_shape=(1, GDN_HEADS)), dtype="f32"
    )
    g = tf.neg(tf.exp(a_log)) * tf.softplus(a_prj + dt_bias)  # (1,32) f32
    g_exp = tf.reshape(tf.exp(g), new_shape=(1, GDN_HEADS, 1, 1))

    # ── 递推（M:325-368，f32；q/k 先 l2norm，q 再乘 scale）────────────
    # l2norm 在 bf16 里做（M:328-331 在 .to(f32) 之前），之后才升 f32
    q_ss = tf.reduce(q * q, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    q = q * tf.rsqrt(q_ss + tf.full_like(q_ss, value=GDN_EPS))
    k_ss = tf.reduce(k * k, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    k = k * tf.rsqrt(k_ss + tf.full_like(k_ss, value=GDN_EPS))
    qf = tf.cast(q, dtype="f32") * tf.full_like(tf.cast(q, dtype="f32"), value=GDN_SCALE)
    kf = tf.cast(k, dtype="f32")
    vf = tf.cast(v, dtype="f32")

    s_decayed = rec_state * g_exp                             # (1,32,128k,128v)
    kv_mem = tf.reduce(
        s_decayed * tf.reshape(kf, new_shape=(1, GDN_HEADS, GDN_HDIM, 1)),
        axes=(2,), keepdim=False, kind=ReduceKind.SUM,
    )                                                         # (1,32,128v)
    delta = (vf - kv_mem) * tf.reshape(beta, new_shape=(1, GDN_HEADS, 1))
    rec_state_new = s_decayed + tf.reshape(kf, new_shape=(1, GDN_HEADS, GDN_HDIM, 1)) * tf.reshape(
        delta, new_shape=(1, GDN_HEADS, 1, GDN_HDIM)
    )
    out = tf.reduce(
        rec_state_new * tf.reshape(qf, new_shape=(1, GDN_HEADS, GDN_HDIM, 1)),
        axes=(2,), keepdim=False, kind=ReduceKind.SUM,
    )                                                         # (1,32,128) f32

    # ── RMSNormGated（M:185-200）: 递推输出先回 bf16（M:366 initial_dtype），
    #    norm 内升 f32 → rsqrt → 回 bf16 × w（原样）→ × silu(z.f32) → bf16 ──
    out_bf = tf.cast(out, dtype="bf16")
    of = tf.cast(out_bf, dtype="f32")
    o_ss = tf.reduce(of * of, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    o_ms = o_ss * tf.full_like(o_ss, value=GDN_INV_HDIM)
    o_n = tf.cast(of * tf.rsqrt(o_ms + tf.full_like(o_ms, value=GDN_EPS)), dtype="bf16")
    o_n = o_n * gated_norm_gamma                              # M:198（权重原样，非 1+w）
    z = tf.cast(
        tf.reshape(tf.matmul(h, tf.transpose(w_z, perm=(1, 0))), new_shape=(1, GDN_HEADS, GDN_HDIM)),
        dtype="f32",
    )
    o_g = tf.cast(o_n, dtype="f32") * (z * tf.sigmoid(z))     # M:199
    o_flat = tf.reshape(tf.cast(o_g, dtype="bf16"), new_shape=(1, 1, GDN_VALUE_DIM))

    # ── out_proj → 残差 ──────────────────────────────────────────────
    y = tf.matmul(o_flat, tf.transpose(w_out, perm=(1, 0)))
    return x + y, conv_state_new, rec_state_new


@func
def gdn_convert(
    input_layernorm: ConstTensor[(HIDDEN,), "f32"],                     # RAW ckpt layernorm 权重（未 +1）
    in_proj_qkv: ConstTensor[(GDN_CONV_DIM, HIDDEN), "bf16"],           # nn.Linear 原生 [out,in]
    in_proj_z: ConstTensor[(GDN_VALUE_DIM, HIDDEN), "bf16"],
    in_proj_b: ConstTensor[(GDN_HEADS, HIDDEN), "bf16"],
    in_proj_a: ConstTensor[(GDN_HEADS, HIDDEN), "bf16"],
    conv1d_weight: ConstTensor[(GDN_CONV_DIM, 1, GDN_CONV_K), "bf16"],  # nn.Conv1d 原生 [C,1,K]
    a_log: ConstTensor[(GDN_HEADS,), "f32"],
    dt_bias: ConstTensor[(GDN_HEADS,), "f32"],
    norm_weight: ConstTensor[(GDN_HDIM,), "bf16"],                      # RMSNormGated 权重，原样
    out_proj: ConstTensor[(HIDDEN, GDN_VALUE_DIM), "bf16"],             # nn.Linear 原生 [out,in]
):
    # 返回 (in_norm_raw, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias,
    # gated_norm_gamma, w_out)——gdn_mix 的 CANONICAL（= kernels/linear_attn.py
    # LinearAttnWeights 原生 layout）权重。唯一 convert 入口：evaluator（经
    # gdn_mix）和 kernel override 都读这一份产物。全部保持 nn.Linear/
    # nn.Conv1d 原生 [out,in]（不转置），norm 保持 RAW（未 +1），conv1d 权重
    # squeeze 掉 groups 维；纯 repack/cast，无数值变换。
    in_norm_raw = tf.cast(input_layernorm, dtype="f32")
    w_qkv = tf.cast(in_proj_qkv, dtype="bf16")
    w_z = tf.cast(in_proj_z, dtype="bf16")
    w_b = tf.cast(in_proj_b, dtype="bf16")
    w_a = tf.cast(in_proj_a, dtype="bf16")
    conv_w = tf.reshape(tf.cast(conv1d_weight, dtype="bf16"), new_shape=(GDN_CONV_DIM, GDN_CONV_K))
    a_log_raw = tf.cast(a_log, dtype="f32")
    dt_bias_raw = tf.cast(dt_bias, dtype="f32")
    gated_norm_gamma = tf.cast(norm_weight, dtype="bf16")
    w_out = tf.cast(out_proj, dtype="bf16")
    return in_norm_raw, w_qkv, w_z, w_b, w_a, conv_w, a_log_raw, dt_bias_raw, gated_norm_gamma, w_out


# ── 模型头尾（root 的 fusion func；权重原生/raw，无需 convert）────────────


@func
def embed(
    token_id: Tensor[(1,), "i32"],
    embed_tokens: ConstTensor[(VOCAB, HIDDEN), "bf16"],
):
    # 残差流起点：按 token id 取 embed 表一行（M:embed_tokens）。
    h = tf.gather(embed_tokens, token_id, axis=0)         # (1, HIDDEN) bf16
    return tf.reshape(h, new_shape=(1, 1, HIDDEN))


@func
def head(
    x: Tensor[(1, 1, HIDDEN), "bf16"],
    final_norm_raw: ConstTensor[(HIDDEN,), "f32"],        # RAW（func 体内 1+w）
    lm_head: ConstTensor[(VOCAB, HIDDEN), "bf16"],        # 原生 [vocab,hidden]（= lm_head_gemv kernel 的 layout）
) -> Tensor[(1, VOCAB), "f32"]:
    # final RMSNorm（M:norm）+ lm_head GEMV → f32 logits（对齐 lm_head_gemv 的
    # f32 累加/输出）。matmul(lm_head[vocab,hidden], h_col[hidden,1]) 避免转置大权重。
    h = tf.rms_norm(x, 1.0 + final_norm_raw)              # (1,1,HIDDEN) bf16
    h_col = tf.reshape(tf.cast(h, dtype="f32"), new_shape=(HIDDEN, 1))
    logits = tf.matmul(tf.cast(lm_head, dtype="f32"), h_col)   # (VOCAB, 1) f32
    return tf.reshape(logits, new_shape=(1, VOCAB))
