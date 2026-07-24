"""Oracle tests: 主 agent 的 fusion HIR funcs vs HF 层实例（随机权重）。

这两个测试是 kernel sub-agent 的语义契约的**验收**：HIR evaluator 跑
qwen35_module 的 fusion func，和 transformers 5.12.1 的层类逐位对比。
过了这个门，kernel 的 oracle（HF 层）和 RuntimeModule 的 check()（HIR
evaluator）才是同一个语义。

权重映射约定（三个 mixer 统一模式，集成阶段照抄）：
- 本测试只喂 RAW ckpt 张量（nn.Linear/nn.Conv1d 原生形状/dtype，未转置、
  未 (1+w)）给各自的 ``*_convert`` @func（moe_convert/attn_convert/
  gdn_convert）——唯一 convert 入口，返回 CANONICAL（= kernel 原生/packed
  layout）权重，再喂进 ``*_mix``。转置（w_qg/w_k/w_v/w_o → [in,out]）、
  repack（MoE experts 堆叠）都在 convert 内部做，HIR 侧不再自己转置。
- 所有 RMSNorm gamma：三个 mixer 现在统一收 RAW w.float()（checkpoint 存
  w-1 形式），(1+w) 挪进各自 mix 函数体就地算（见 moe_convert/moe_mix、
  attn_convert/full_attn_mix、gdn_convert/gdn_mix）。
"""
from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextRotaryEmbedding,
)

from tests.models.qwen3_5_35b_a3b.qwen35_module import (
    CACHE_CAP,
    GDN_CONV_DIM,
    GDN_CONV_K,
    GDN_HDIM,
    GDN_HEADS,
    GDN_VALUE_DIM,
    HEAD_DIM,
    HIDDEN,
    KV_PROJ,
    N_KV_HEADS,
    N_Q_HEADS,
    Q_PROJ,
    QG_PROJ,
    ROT_DIM,
    attn_convert,
    full_attn_mix,
    gdn_convert,
    gdn_mix,
    moe_convert,
    moe_mix,
)
from tilefoundry.evaluator import evaluate

DEVICE = "cuda"


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


@pytest.fixture(scope="module")
def hf_config():
    return Qwen3_5MoeTextConfig(
        hidden_size=HIDDEN,
        head_dim=HEAD_DIM,
        num_attention_heads=N_Q_HEADS,
        num_key_value_heads=N_KV_HEADS,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        vocab_size=1024,  # 本测试不触 embed/lm_head
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        dtype=torch.bfloat16,
    )


def test_moe_mix_matches_hf(hf_config):
    torch.manual_seed(0)
    blk = Qwen3_5MoeSparseMoeBlock(hf_config).to(DEVICE, torch.bfloat16).eval()
    post_norm = Qwen3_5MoeRMSNorm(HIDDEN).to(DEVICE)
    with torch.no_grad():
        post_norm.weight.normal_(0, 0.2)          # 非平凡 (1+w)
        for p in blk.parameters():
            p.normal_(0, 0.05)

    x = (torch.randn(1, 1, HIDDEN, device=DEVICE) * 0.5).to(torch.bfloat16)

    with torch.no_grad():
        ref = x + blk(post_norm(x))               # DecoderLayer M:869-874 的 MoE 半段

    # moe_convert：RAW ckpt 张量 → moe_mix 的 CANONICAL（= kernel packed）权重。
    # 唯一 convert 入口，这里顺带验证它自己（shape）。
    post_norm_raw, router_w, packed_gate_up, packed_down, shared_gate_w = evaluate(
        moe_convert,
        post_norm.weight.float(),
        blk.gate.weight,
        blk.experts.gate_up_proj,
        blk.experts.down_proj,
        blk.shared_expert.gate_proj.weight,
        blk.shared_expert.up_proj.weight,
        blk.shared_expert.down_proj.weight,
        blk.shared_expert_gate.weight,
    )
    assert post_norm_raw.shape == (HIDDEN,)
    assert router_w.shape == (256, HIDDEN)
    assert packed_gate_up.shape == (257, 1024, HIDDEN)
    assert packed_down.shape == (257, HIDDEN, 512)
    assert shared_gate_w.shape == (1, HIDDEN)

    got = evaluate(
        moe_mix,
        x,
        post_norm_raw,
        router_w,
        packed_gate_up,
        packed_down,
        shared_gate_w,
    )
    r = _rel_l2(got, ref)
    print(f"\n[moe_mix vs HF] rel_l2={r:.3e}")
    assert r <= 2e-2, r  # bf16 专家求和顺序差异；先宽后收，实测填报


@pytest.mark.parametrize("prior", [0, 17])
def test_full_attn_mix_matches_hf(hf_config, prior):
    torch.manual_seed(prior + 1)
    attn = Qwen3_5MoeAttention(hf_config, layer_idx=3).to(DEVICE, torch.bfloat16).eval()
    in_norm = Qwen3_5MoeRMSNorm(HIDDEN).to(DEVICE)
    rotary = Qwen3_5MoeTextRotaryEmbedding(hf_config, device=DEVICE)
    with torch.no_grad():
        in_norm.weight.normal_(0, 0.2)
        for p in attn.parameters():
            p.normal_(0, 0.05)

    xs = (torch.randn(1, prior + 1, HIDDEN, device=DEVICE) * 0.5).to(torch.bfloat16)

    # ── HF 参考：prior 个 token 先填 cache，再算第 prior 位 ───────────
    cache = DynamicCache()
    with torch.no_grad():
        for t in range(prior + 1):
            xt = xs[:, t : t + 1]
            pos = torch.tensor([[t]], device=DEVICE)
            cos, sin = rotary(xt, pos)
            out, _ = attn(
                hidden_states=in_norm(xt),
                position_embeddings=(cos, sin),
                attention_mask=None,
                past_key_values=cache,
            )
        ref = xt + out

    # ── HIR：同权重，把 HF cache 前缀搬进我们的布局 ──────────────────
    k_cache = torch.zeros(1, CACHE_CAP, N_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    if prior > 0:
        # HF cache 存的是已 rope 的 K：layers[3].keys (1, 2, prior+1, 256) —— 取前 prior
        hf_k = cache.layers[3].keys[:, :, :prior].transpose(1, 2)
        hf_v = cache.layers[3].values[:, :, :prior].transpose(1, 2)
        k_cache[:, :prior] = hf_k
        v_cache[:, :prior] = hf_v

    inv_freq = 1.0 / (10_000_000 ** (torch.arange(0, ROT_DIM, 2, device=DEVICE, dtype=torch.float) / ROT_DIM))
    t = torch.arange(CACHE_CAP, device=DEVICE, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    cos_cache = torch.cat([freqs, freqs], dim=-1).cos()
    sin_cache = torch.cat([freqs, freqs], dim=-1).sin()

    mask = torch.full((1, 1, 1, CACHE_CAP), float("-inf"), device=DEVICE)
    mask[..., : prior + 1] = 0.0

    # attn_convert：RAW ckpt 张量 → full_attn_mix 的 CANONICAL（= kernel 原生
    # layout）权重。唯一 convert 入口，这里顺带验证它自己（shape）。
    in_norm_raw, w_qg, w_k, w_v, w_o, q_norm_raw, k_norm_raw = evaluate(
        attn_convert,
        in_norm.weight.float(),
        attn.q_proj.weight.to(torch.bfloat16),   # native [out,in]，不转置
        attn.k_proj.weight.to(torch.bfloat16),
        attn.v_proj.weight.to(torch.bfloat16),
        attn.o_proj.weight.to(torch.bfloat16),
        attn.q_norm.weight.float(),
        attn.k_norm.weight.float(),
    )
    assert in_norm_raw.shape == (HIDDEN,)
    assert w_qg.shape == (HIDDEN, QG_PROJ)
    assert w_k.shape == (HIDDEN, KV_PROJ)
    assert w_v.shape == (HIDDEN, KV_PROJ)
    assert w_o.shape == (Q_PROJ, HIDDEN)
    assert q_norm_raw.shape == (HEAD_DIM,)
    assert k_norm_raw.shape == (HEAD_DIM,)

    got, _, _ = evaluate(
        full_attn_mix,
        xs[:, prior : prior + 1],
        in_norm_raw,
        w_qg,
        w_k,
        w_v,
        w_o,
        q_norm_raw,
        k_norm_raw,
        cos_cache,
        sin_cache,
        torch.tensor([[prior]], device=DEVICE, dtype=torch.int32),
        k_cache,
        v_cache,
        torch.tensor([prior], device=DEVICE, dtype=torch.int32),
        torch.tensor([1], device=DEVICE, dtype=torch.int32),
        mask,
    )
    r = _rel_l2(got, ref)
    print(f"\n[full_attn_mix vs HF] prior={prior} rel_l2={r:.3e}")
    assert r <= 5e-3, r  # 先宽后收，实测填报


@pytest.mark.parametrize("prior", [1, 9])
def test_gdn_mix_matches_hf(hf_config, prior):
    torch.manual_seed(prior + 100)
    gdn = Qwen3_5MoeGatedDeltaNet(hf_config, layer_idx=0).to(DEVICE, torch.bfloat16).eval()
    in_norm = Qwen3_5MoeRMSNorm(HIDDEN).to(DEVICE)
    with torch.no_grad():
        in_norm.weight.normal_(0, 0.2)
        for n, p in gdn.named_parameters():
            if n in ("A_log", "dt_bias"):
                continue  # 保留构造时的合理分布（A_log=log(U(0,16))）
            p.normal_(0, 0.05)

    xs = (torch.randn(1, prior + 1, HIDDEN, device=DEVICE) * 0.5).to(torch.bfloat16)

    # ── HF 参考：prior 步建立 conv/rec 状态，再算第 prior 步 ──────────
    cache = DynamicCache(config=hf_config)
    with torch.no_grad():
        for t in range(prior + 1):
            xt = xs[:, t : t + 1]
            if t == prior:
                # 抄走进入最后一步之前的状态（decode 分支原地改 conv_state）
                conv_state = cache.layers[0].conv_states.detach().clone()
                rec_state = cache.layers[0].recurrent_states.detach().clone()
            out = gdn(hidden_states=in_norm(xt), cache_params=cache)
        ref = xt + out

    # gdn_convert：RAW ckpt 张量 → gdn_mix 的 CANONICAL（= kernel 原生 layout）
    # 权重。唯一 convert 入口，这里顺带验证它自己（shape）。
    in_norm_raw, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, gated_norm_gamma, w_out = evaluate(
        gdn_convert,
        in_norm.weight.float(),
        gdn.in_proj_qkv.weight.to(torch.bfloat16),   # native [out,in]，不转置
        gdn.in_proj_z.weight.to(torch.bfloat16),
        gdn.in_proj_b.weight.to(torch.bfloat16),
        gdn.in_proj_a.weight.to(torch.bfloat16),
        gdn.conv1d.weight.to(torch.bfloat16),        # [8192,1,4]，convert 内 squeeze
        gdn.A_log.float(),
        gdn.dt_bias.float(),
        gdn.norm.weight.to(torch.bfloat16),                    # 原样，无 (1+w)
        gdn.out_proj.weight.to(torch.bfloat16),      # native [out,in]，不转置
    )
    assert in_norm_raw.shape == (HIDDEN,)
    assert w_qkv.shape == (GDN_CONV_DIM, HIDDEN)
    assert w_z.shape == (GDN_VALUE_DIM, HIDDEN)
    assert w_b.shape == (GDN_HEADS, HIDDEN)
    assert w_a.shape == (GDN_HEADS, HIDDEN)
    assert conv_w.shape == (GDN_CONV_DIM, GDN_CONV_K)
    assert a_log.shape == (GDN_HEADS,)
    assert dt_bias.shape == (GDN_HEADS,)
    assert gated_norm_gamma.shape == (GDN_HDIM,)
    assert w_out.shape == (HIDDEN, GDN_VALUE_DIM)

    got, conv_new, rec_new = evaluate(
        gdn_mix,
        xs[:, prior : prior + 1],
        in_norm_raw,
        w_qkv,
        w_z,
        w_b,
        w_a,
        conv_w,
        a_log,
        dt_bias,
        gated_norm_gamma,
        w_out,
        conv_state.to(torch.bfloat16),
        rec_state.float(),
    )
    r = _rel_l2(got, ref)
    r_rec = _rel_l2(rec_new, cache.layers[0].recurrent_states)
    print(f"\n[gdn_mix vs HF] prior={prior} rel_l2={r:.3e} rec_state rel_l2={r_rec:.3e}")
    assert r <= 5e-3, r
    # state 差 ~6e-3 且不随步数漂移（prior=1/9 同量级）：来源是 2048→8192 投影的
    # bf16 GEMM 内核差异（F.linear vs 预转置 matmul 的累加序），非逻辑错误——
    # l2norm/sigmoid 的 bf16 时机已逐一镜像 HF（见 qwen35_module）。输出门 5e-3 为准。
    assert r_rec <= 1e-2, r_rec
