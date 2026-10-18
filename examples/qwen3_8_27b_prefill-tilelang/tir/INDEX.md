# The finalized TIR, and why 35 files stand for 75

`tools/finalize_all.sh` lowers every scheduled HIR function to TIR, one file per
(function, prompt length): **75 files**. Only **35 of them are different
programs**. The other 40 are the same lowering with other extents, so this
directory keeps one representative of each and this table says where every one
of the 75 went. Regenerate the full set with `tools/finalize_all.sh`.

Two files count as the same program after removing all generated comment lines
(including analysis, selection, memory, buffer, reuse and error comments), blanking
integer literals, and treating `ConstTensor` as `Tensor` only for read-only input
parameters. Read-only inputs are checked against instruction memory effects through
their views. Parameter ownership annotations do not change the lowered body. Every
other text difference denotes a different program. Rows explicitly described as
byte for byte require byte equality without normalization.

**The redundancy is the segmentation result, read off the files.**

| | |
|---|---|
| 11 functions are the same at all three lengths | `embed`, `final`, `rmsnorm`, `gated_norm`, `prep_*` (5), `delta_rule_*` (3). 33 files, 11 programs |
| the GEMMs split 1024 from 4096/8192, and nowhere else | `attention`, `qkv_whole`, `in_proj_whole`, `gate_up_whole`, `down_whole`: 4096 and 8192 are the same program, 1024 is its own |
| three GEMMs share one lowering | `down_whole`, `o_proj_whole`, `out_proj_whole` are one program at each length — `o_proj_whole_8192.py` and `out_proj_whole_8192.py` are byte for byte identical |
| `qkv` at 8192 is the exception | `qkv_pieces`, `qkv_tail`, `qkv_whole_last` and `qkv_whole_b` exist only where the tail plan needs them |

The boundary argued in `README.md` §5 is **1024 against 4096** — and that is
exactly where, and the only place where, the lowered code changes.

## Every file, and which one stands for it

| lowered (function, length) | in this directory |
|---|---|
| `attention_1024.py` | **kept** |
| `attention_4096.py` | **kept** |
| `attention_8192.py` | `attention_4096.py` with other extents |
| `delta_rule_intra_g_1024.py` | **kept** |
| `delta_rule_intra_g_4096.py` | `delta_rule_intra_g_1024.py` with other extents |
| `delta_rule_intra_g_8192.py` | `delta_rule_intra_g_1024.py` with other extents |
| `delta_rule_intra_wup_1024.py` | **kept** |
| `delta_rule_intra_wup_4096.py` | `delta_rule_intra_wup_1024.py` with other extents |
| `delta_rule_intra_wup_8192.py` | `delta_rule_intra_wup_1024.py` with other extents |
| `delta_rule_recur_out_1024.py` | **kept** |
| `delta_rule_recur_out_4096.py` | `delta_rule_recur_out_1024.py` with other extents |
| `delta_rule_recur_out_8192.py` | `delta_rule_recur_out_1024.py` with other extents |
| `down_pieces_1024.py` | **kept** |
| `down_tail_1024.py` | **kept** |
| `down_whole_1024.py` | **kept** |
| `down_whole_4096.py` | **kept** |
| `down_whole_8192.py` | `down_whole_4096.py` with other extents |
| `embed_1024.py` | **kept** |
| `embed_4096.py` | `embed_1024.py` with other extents |
| `embed_8192.py` | `embed_1024.py` with other extents |
| `final_1024.py` | **kept** |
| `final_4096.py` | `final_1024.py` with other extents |
| `final_8192.py` | `final_1024.py` with other extents |
| `gate_up_pack.py` | **kept** |
| `gate_up_pieces_1024.py` | **kept** |
| `gate_up_tail_1024.py` | **kept** |
| `gate_up_whole_1024.py` | **kept** |
| `gate_up_whole_4096.py` | **kept** |
| `gate_up_whole_8192.py` | `gate_up_whole_4096.py` with other extents |
| `gated_norm_1024.py` | **kept** |
| `gated_norm_4096.py` | `gated_norm_1024.py` with other extents |
| `gated_norm_8192.py` | `gated_norm_1024.py` with other extents |
| `in_proj_pack.py` | **kept** |
| `in_proj_whole_1024.py` | **kept** |
| `in_proj_whole_4096.py` | **kept** |
| `in_proj_whole_8192.py` | `in_proj_whole_4096.py` with other extents |
| `o_proj_pieces_1024.py` | `down_pieces_1024.py` with other extents |
| `o_proj_tail_1024.py` | identical to `down_tail_1024.py` |
| `o_proj_whole_1024.py` | `down_whole_1024.py` with other extents |
| `o_proj_whole_4096.py` | `down_whole_4096.py` with other extents |
| `o_proj_whole_8192.py` | `down_whole_4096.py` with other extents |
| `out_proj_pieces_1024.py` | `down_pieces_1024.py` with other extents |
| `out_proj_tail_1024.py` | identical to `down_tail_1024.py` |
| `out_proj_whole_1024.py` | `down_whole_1024.py` with other extents |
| `out_proj_whole_4096.py` | `down_whole_4096.py` with other extents |
| `out_proj_whole_8192.py` | `down_whole_4096.py` with other extents |
| `prep_conv_k_1024.py` | **kept** |
| `prep_conv_k_4096.py` | `prep_conv_k_1024.py` with other extents |
| `prep_conv_k_8192.py` | `prep_conv_k_1024.py` with other extents |
| `prep_conv_q_1024.py` | **kept** |
| `prep_conv_q_4096.py` | `prep_conv_q_1024.py` with other extents |
| `prep_conv_q_8192.py` | `prep_conv_q_1024.py` with other extents |
| `prep_conv_v_1024.py` | **kept** |
| `prep_conv_v_4096.py` | `prep_conv_v_1024.py` with other extents |
| `prep_conv_v_8192.py` | `prep_conv_v_1024.py` with other extents |
| `prep_gate_beta_1024.py` | **kept** |
| `prep_gate_beta_4096.py` | `prep_gate_beta_1024.py` with other extents |
| `prep_gate_beta_8192.py` | `prep_gate_beta_1024.py` with other extents |
| `prep_gate_g_1024.py` | **kept** |
| `prep_gate_g_4096.py` | `prep_gate_g_1024.py` with other extents |
| `prep_gate_g_8192.py` | `prep_gate_g_1024.py` with other extents |
| `qkv_pack.py` | **kept** |
| `qkv_pieces_1024.py` | `down_pieces_1024.py` with other extents |
| `qkv_pieces_8192.py` | **kept** |
| `qkv_tail_1024.py` | **kept** |
| `qkv_tail_8192.py` | **kept** |
| `qkv_whole_1024.py` | **kept** |
| `qkv_whole_4096.py` | **kept** |
| `qkv_whole_8192.py` | `qkv_whole_4096.py` with other extents |
| `qkv_whole_b_8192.py` | **kept** |
| `qkv_whole_last_1024.py` | **kept** |
| `qkv_whole_last_8192.py` | **kept** |
| `rmsnorm_1024.py` | **kept** |
| `rmsnorm_4096.py` | `rmsnorm_1024.py` with other extents |
| `rmsnorm_8192.py` | `rmsnorm_1024.py` with other extents |

## Current regeneration status

All **75 of 75** jobs completed, and all **35 groups** were regenerated.
Every generated file passed TIR verification (`verify_module`). All 75-to-35
normalized comparisons match the table above. The o_proj/out_proj 8192 pair and
both projection-tail/down-tail 1024 pairs remain byte for byte identical.

The two QKV whole jobs at 4096 and 8192 completed successfully after about
42 minutes each. They retain native exact ISL counting; these cases remain
expensive rather than substituting an approximate count.
