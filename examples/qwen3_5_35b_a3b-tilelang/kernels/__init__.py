"""tilelang kernels for the Qwen3.5-35B-A3B decode step.

Organisation
------------
One module per Module boundary in `model.py`, plus `basic` for the primitives
every boundary shares:

    basic   rms_norm / gemv / residual_add / embed / lm_head -- the shapes that
            appear in more than one place
    gdn     Qwen3_5LinearAttention: conv_step, l2_normalise, delta_step, and the
            fused whole-mixer step
    attn    Qwen3_5FullAttention: partial_rope, partial_rope_kv, and the fused
            whole-mixer step
    moe     Qwen3_5Router + Qwen3_5MoE: routing, routed_experts, shared_expert
    torch_ref  the same functions in plain torch

`torch_ref` is not dead weight: every kernel here is selectable per function
through `TF_IMPL` (see `runtime_model.py`), which is what makes a wrong output
bisectable to one kernel instead of one step. A decode step calls ~250 kernels;
without that, "the logits are wrong" names all of them.

Every kernel in this package is decode-shaped: sequence length is 1 and the
matmuls are therefore GEMV. That is a real specialisation, not a simplification
-- `model.py` declares `S = 1`, so a batched implementation would be describing a
different Module.
"""
