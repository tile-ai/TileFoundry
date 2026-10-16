# Issue 237 validation and changed artifacts

Base: upstream `aaa8054d` (#238). Full verification commands use `RUN=issue237-m8-final-source tilefoundry-verify source` and `RUN=issue237-m8-final-installed tilefoundry-verify installed`.

## Final artifact regeneration

- Long: 69/75 jobs completed; D70 comparisons preserve 35 groups. 31 complete groups regenerated; four representatives retain the committed artifact because six jobs remain in upstream ISL `count_val` timeouts. `examples/qwen3_8_27b_prefill-tilelang/tir/INDEX.md` lists every retained representative and blocked job. The o_proj/out_proj 8192 byte comparison passes.
- Short: four generators used the unchanged TileLang environment with this checkout on PYTHONPATH. All 64 outputs are checked with run_path and TIR verification.
- down_pieces_1024 cache capacity diagnostics match aaa8054d memory analysis on the identical HIR: w/x both hold 84 MB, exceed 47.68 MB, fits=no. They are upstream diagnostics, not a new finding. Baseline finalize refuses the old dynamic staged loop; memory analysis provides the comparison without lowering.
- scheduled_index table matches BENCH section 7; chain_table regenerates BENCH section 10. GPU measurements are unchanged. The four refreshed short composed roofline numbers and traffic/FLOP totals match aaa8054d (D53); see the short MIGRATION.md table.

## Address and allocation ledger

- 180 reports: 140 byte-identical to aaa8054d; 40 changed. All 180 exactly match the accepted D65 reports. Memory aliases/offsets/peaks follow D50/D51/D56; wide MN-major Wgmma admission follows D18/D30; scalar ceildiv follows D32; sparse_decode changes follow D62/D63.
- 11 placed records: all equal the D65 records. Ten differ from aaa8054d through the accepted D50/D51 semantic same-address publication and semantic-first priority; the Qwen allocation peak includes the attributed +4096 bytes.
- IN_PLACE: all 156 records exactly equal D65. The original D57 baseline had 153; the three additional records follow the D63 sparse_decode fixture rewrite. D57 behavior remains unchanged.
- Reshard/Bitcast: 280 Reshards, 67 aliases, 7 Bitcasts; the 281/69/8 to 280/67/7 change is D62/D63, including removal of the compensating ks_view Reshard.
- Address audit: all 48 current TIR goldens were compared against aaa8054d, expanding pointer aliases and normalizing allocation/loop names. Existing instruction origin expressions agree except sparse_decode (D63 removes compensating permutations). New SSA views preserve the exact Summary address relation (D58-D65); output ownership follows D66. The 16 new goldens have recorded address expressions.

## Changed test goldens

| File | Cause |
|---|---|
| `tests/fixtures/inspection/type_printer_sugar.analyzed.txt` | M6: printer round-trip |
| `tests/fixtures/schedule/Wgmma.facts.txt` | M5 / D18-D31: Wgmma facts/admission and candidate pruning |
| `tests/fixtures/schedule/plain/gemm_8192x17408x5120_cta_grid.candidates.txt` | M5 / D18-D31: Wgmma facts/admission and candidate pruning |
| `tests/fixtures/schedule/tir/captured_insert_update.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/child_weight.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/cta_submesh_thread_scope.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/fp8_block_scaled_gemm.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/gemm_8192x17408x5120_register_store.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/gemm_8192x17408x5120_tma_store.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/inplace_output_update.py` | D50-D52, D58, D66: output-param writes |
| `tests/fixtures/schedule/tir/loop_carry_from_param.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/loop_carry_rebind.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/loop_carry_window.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/loop_outputs.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/loop_variable_rebound.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/mesh_capture_outside.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/mesh_views.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/mesh_zeros_submesh.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/module_instances.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/register_arange.py` | M4 / D10: T.arange |
| `tests/fixtures/schedule/tir/register_full_like.py` | M4: typed fill |
| `tests/fixtures/schedule/tir/register_index_window.py` | M4 / D33, D60: register window origin |
| `tests/fixtures/schedule/tir/register_index_window_offset.py` | M4 / D33, D60: offset register window |
| `tests/fixtures/schedule/tir/reshard_views.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/scalar_binary.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/scalar_ceildiv_gate.py` | M6 / D32: value ceildiv |
| `tests/fixtures/schedule/tir/sm80_mma_ldmatrix.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/sparse_decode.py` | D62/D63: row-major mapping and canonical fixture |
| `tests/fixtures/schedule/tir/staged_dynamic_inner.py` | D68: dynamic stage counter |
| `tests/fixtures/schedule/tir/staged_nested.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/strided_return_window.py` | D61: exact strided view |
| `tests/fixtures/schedule/tir/submesh_slice_in_sibling.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_a_k_major.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_a_mn_major.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_b_mn_major_grouped_n64.py` | D47: preserve broadcast operand mode structure |
| `tests/fixtures/schedule/tir/wgmma_b_mn_major_wide_lbo.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_b_nk.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_cast_between_schedules.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_cp_async_loads.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_cta_grid_4x17.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_explicit_windows.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_k_slices_of_wide_run.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_one_tile_of_larger_output.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_repeat_along_k.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_repeat_along_n_order.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_rs_a_from_accumulator.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_rs_a_from_smem.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_swizzled_smem.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_tma_3stage.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/wgmma_two_schedules.py` | D58-D66: distinct SSA views at analyzed storage; declaration/naming changes |
| `tests/fixtures/schedule/tir/zero_trip_insert_slice.py` | D64: accepted direct address proof |

## Changed example artifacts

M7/D35/D45/D46 migrate explicit destinations and canonical layout order; D47/D58-D68 allow the original program structures to lower. The regenerated artifact files are:

- `examples/qwen3_8_27b_prefill-tilelang/tir/attention_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/attention_4096.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/delta_rule_intra_g_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/delta_rule_intra_wup_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/delta_rule_recur_out_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/down_tail_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/down_whole_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/down_whole_4096.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/embed_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/final_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/gate_up_pack.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/gate_up_tail_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/gate_up_whole_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/gate_up_whole_4096.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/gated_norm_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/in_proj_pack.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/in_proj_whole_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/in_proj_whole_4096.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/prep_conv_k_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/prep_conv_q_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/prep_conv_v_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/prep_gate_beta_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/prep_gate_g_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/qkv_pack.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/qkv_tail_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/qkv_tail_8192.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/qkv_whole_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/qkv_whole_b_8192.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/qkv_whole_last_1024.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/qkv_whole_last_8192.py`
- `examples/qwen3_8_27b_prefill-tilelang/tir/rmsnorm_1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/attention/attn_core_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/attention/attn_core_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/attention/attn_core_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/attention/attn_core_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/attention/attn_prep_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/attention/attn_prep_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/attention/attn_prep_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/attention/attn_prep_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_chunk_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_chunk_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_chunk_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_chunk_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_norm_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_norm_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_norm_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_norm_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_prep_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_prep_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_prep_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_prep_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_scan_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_scan_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_scan_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gdn/gdn_scan_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/att_in_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/att_in_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/att_in_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/att_in_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/down_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/down_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/down_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/down_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gate_up_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gate_up_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gate_up_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gate_up_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gdn_ba_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gdn_ba_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gdn_ba_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gdn_ba_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gdn_in_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gdn_in_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gdn_in_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/gdn_in_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/o_proj_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/o_proj_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/o_proj_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/gemm/o_proj_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/embed_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/embed_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/embed_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/embed_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/lm_head.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/reduce_ba_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/reduce_ba_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/reduce_ba_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/reduce_ba_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/reduce_sk2_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/reduce_sk2_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/rmsnorm_T1024.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/rmsnorm_T128.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/rmsnorm_T256.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/rmsnorm_T512.py`
- `examples/qwen3_8_27b_prefill_short-tilelang/tir/small/rmsnorm_last.py`

## Source net and retained behavior

Positive lines add required behavior; simplifications share A/B Wgmma patterns, remove structural output-root tracing and duplicate layout/address enumeration, replace old binder bookkeeping, and unify optional reuse. No features were removed to reduce lines.

| Source file | Added minus deleted | Reason |
|---|---:|---|
| `src/tilefoundry/analysis/access.py` | +0 | Address relation write tracking / optional reuse representation |
| `src/tilefoundry/analysis/allocation.py` | +106 | D50-D65 semantic address facts, exact executable relations and generic IN_PLACE summaries |
| `src/tilefoundry/analysis/memory.py` | +1 | Do not charge material bytes to semantic aliases |
| `src/tilefoundry/analysis/visitor.py` | +0 | Use the upstream analysis scope context |
| `src/tilefoundry/cli/__init__.py` | +0 | Correct CLI help |
| `src/tilefoundry/inspection/pattern_printer.py` | -3 | One shared operator/precedence table |
| `src/tilefoundry/inspection/printer_base.py` | +0 | One NaN spelling with import context |
| `src/tilefoundry/inspection/python_printer.py` | +0 | Propagate print context to importable values |
| `src/tilefoundry/inspection/tir_printer.py` | +0 | Reuse TIR print context for constants |
| `src/tilefoundry/ir/hir/schedule.py` | +25 | D19 sparse candidate admission and concrete pruning |
| `src/tilefoundry/ir/hir/sharding/local.py` | -12 | Reuse the true local-layout projection |
| `src/tilefoundry/ir/hir/sharding/reshard.py` | +4 | D38/D39 storage-instance equality |
| `src/tilefoundry/ir/hir/tensor/bitcast.py` | +10 | Shared enumeration and row-major correspondence (D62) |
| `src/tilefoundry/ir/hir/tensor/reshape.py` | +6 | Preserve view layout for unit removal |
| `src/tilefoundry/ir/pattern/utils.py` | +14 | D19 field admission reused by call checking |
| `src/tilefoundry/ir/tir/arange.py` | +50 | New TIR Arange op, inference and evaluator |
| `src/tilefoundry/ir/tir/cuda/nn/wgmma.py` | -31 | Merge symmetric A/B shared-memory factories |
| `src/tilefoundry/ir/types/layout_correspondence.py` | -13 | Delete the crossing-axis coordinate fallback |
| `src/tilefoundry/ir/types/shard_layout.py` | +21 | Canonical split order / residual checks, symbolic pass-through |
| `src/tilefoundry/parser/pattern_nodes.py` | +31 | Value-role ceildiv grammar using existing Dim nodes |
| `src/tilefoundry/passes/transforms/convert_hir_to_tir.py` | -11 | Summary-driven views/output writes, typed fills, dynamic stage counter; delete structural root tracing |
| `src/tilefoundry/utils/isl_utils.py` | +2 | Exact single-piece affine extraction support |
| `src/tilefoundry/visitor_registry/candidates/common.py` | +10 | Arange candidate callback |
| `src/tilefoundry/visitor_registry/contexts.py` | +0 | Keep shared candidate context annotation correct |
| `src/tilefoundry/visitor_registry/instruction.py` | +8 | Generic field admission and single-issue broadcast projection |
| `src/tilefoundry/visitor_registry/op_cost.py` | +1 | Charge actual Reshard moves; zero traffic for aliases |
| `src/tilefoundry/visitor_registry/typeinfer.py` | +14 | Uniform axis-group validation |
| `src/tilefoundry/visitor_registry/verify.py` | +39 | Lexical binder-scope traversal including variants |

Total src net: **+272**. The positive total is reported under D37; it is retained to preserve the required fixes.

## Remaining risks

- Upstream count_val jobs not regenerated: `OutProj1024.pieces`, `QKV4096.whole`, `QKV8192.pieces`, `QKV8192.whole`, `OProj1024.pieces`, `GateUp1024.pieces`. Earlier baseline 300-second timeouts also included all three Attention lengths; those completed in this regeneration.
- The window.py shard-window path can still lose unit modes for genuinely sharded operands (D47).
- Device-level gmem Reshard comparisons still enumerate elements and units in Python (D39).

## Full verification follow-up

The single full source run reported 4228 passed, 12 skipped and two stale-golden mismatches. D61 regenerated views coalesce only compact modes: sparse_decode `(4,16):(16,1)` becomes `64:1` and `(8,8):(512,64)` becomes `64:64`; wide_lbo `(2,8):(512,64)` becomes `16:64`. These are exact equal addresses with unchanged instructions, copies, loops and stages. The two goldens were updated and checked with focused finalize tests.
