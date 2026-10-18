# Analysis and lowering placement validation

Baseline: `fdee139e`. The final implementation follows the revised raw-decision
boundary and static instruction emission design. The original plan's byte-identical
TIR requirement is superseded by nested views, retained IR mesh names, and static
issue points. The analysis source tables and `iteration_scope.py` are unchanged.

## Design choices

| Component | Responsibility |
|---|---|
| `AllocationSummary` | Accepted direct links and unchanged address decisions; no executable alias publication |
| `StoragePlan` | Interpret those decisions, direct op relations, output positions, declaration ownership, and refusals |
| `IssuePlan` | Unique automatic instruction, shared projection, ordered static issue points |
| `LoweringContext` | Lexical context, the plans, names, caches and builder |
| `TirBuilder` | Declare storage, materialize views and append TIR statements |
| `Lowering` | Traverse HIR and dispatch the instruction op to one emission primitive |

- Keep native ISL counting for non-box images. Dropping divisions or replacing
  an image with its bounding box can change the count and is not used.
- Keep allocation links as ordered `IntEnum` decisions; do not merge ties into
  flags or add a new analysis address record. Optional register IN_PLACE reuse
  remains a peak estimate, while semantic carry/window/forward ties share storage.
- Derive a view from its immediate partner's declared relation. Do not compose
  lowering addresses back to a root or use an estimator's projected source map.
- Passing a decision through an intervening chain, or binding a written value
  through a returned view, requires identity coordinates. Unsupported inverse
  relations raise `LoweringError` naming the value and relation. A returned
  reshape of a written value is therefore refused rather than silently misaddressed.
- Keep plans independent of the visitor. `runtime_values` lists HIR inputs;
  the visitor lowers them before handing values to the builder.
- Candidates, schedule inference, and lowering use the same projection. Only
  candidates/inference perform admission. Field/mesh rejection precedes projection
  so candidate diagnostics retain their established order.
- Read parallel binding from the local operand types. Expand each unit's serial
  issues in `order`; do not recreate mesh frames or generate `For(o_*)` loops.
- A sliced pointer holds its complete displacement. The sole `StoragePlan.window`
  rule reduces the layout offset modulo `1 << (bits + base + shift)` using the
  Swizzle fields directly; that offset is relative to the sliced pointer.

## Verification

| Check | Result |
|---|---|
| M7 and migrated boundary tests | 678 passed |
| Final source | 4327 passed, 12 skipped |
| Final installed wheel | 161 passed |
| Pre-commit | All hooks passed |
| Final successful fixture TIR verification | 50 / 50 passed |
| Duplicate output fixture | Original refusal retained |
| Baseline successful definitions | 52 definitions, 1416 leaves; all physical address expressions equal to M5 |
| Views and instruction operands | Exact symbolic comparison equal, including staged selects, arbitrary loop/runtime symbols and swizzles |
| Expanded instruction stream | 330 statements, 640 nonscalar tensor operands equal to M5 |
| Maximum serial issues before implementation | 4 in fixtures; 16 in examples; 2825 ScheduleOp observations |
| Analysis inventory | 80 concrete cases in both baseline and current inventory (the plan named 77) |
| Analysis elapsed time | 60.63 s / 58.29 s = 104.0% of baseline |
| Long example regeneration | 75 / 75 regenerated and TIR-verified; all 35 normalized groups and 3 byte-equality pairs preserved |
| Tutorials | All notebooks executed; schedule output reflects the authorized TIR changes |

Commands used for the final suites:

```bash
RUN=final-source-20261018-04 tilefoundry-verify source
RUN=final-installed-20261018-04 tilefoundry-verify installed
python -m tests.schedule.test_storage_context --write
python scripts/render_tutorial.py
JOBS=6 bash examples/qwen3_8_27b_prefill-tilelang/tools/finalize_all.sh
```

The long-example generator completed 73 jobs in the batch; the two QKV whole
jobs at 4096 and 8192 were then rerun on the frozen final source and both exited
successfully. Each still required about 42 minutes of native exact ISL counting.
All 75 generated TIR files passed `verify_module`, including those last two;
the 35 representative groups and required byte-equality pairs were checked on
the complete generated set. Only the 35 representatives are tracked. No bounding
box or division-dropping approximation replaces the remaining expensive counts.

The golden records at `tests/fixtures/schedule/storage/` state every allocated
leaf's partner, accepted kinds/ties, offsets and refusal. The schedule TIR goldens
exercise both the printed program and TIR verification. The #241 cases
`inplace_output_update`, `zero_trip_insert_slice`, `strided_return_window` and
`loop_carry_window` retain their physical addresses; text changes follow the table
below. No CUDA implementation of pointer slicing is added; the existing #237
limitation remains outside this change.

## M7 behavior and numeric differences

| Change | Evidence and result |
|---|---|
| Mesh union counting | Fix selected parameter fibers before projection, union/coalesce, then count; oversized/unbounded boxes retain existential projection |
| Persistent split-K witness | 33-way raster, 128 CTAs; 9,240,576-byte working set; complete JSON equal to the old counting path |
| Counting time for that witness | 1.95 s versus 10.49 s for the old projection path |
| Empty loop selection | `_first_loops` and `_at_first_iteration` return directly when no loop is selected; avoid redundant quantified zero-dimensional `lexmin` |
| Unit mode preservation | The new MN-major B-window fixture fails WGMMA matching before the fix and finalizes afterward |
| Nested local shape | Split positions use flattened modes, so local type extraction flattens nested layout shape before indexing |
| Reshard | Compare coalesced physical boxes and independent zero-stride Split descriptors; no element/unit enumeration or ownership comparison |

The revised Reshard contract intentionally changes the following analysis cases.
All other placement/root/offset/footprint/reuse/peak records match the baseline.
Fourteen additional reports only lose empty MemoryMetadata loop rows that had
existed solely to publish executable aliases; their numeric records are unchanged.

| Case | Difference caused by structural Reshard aliasing |
|---|---|
| `FlashSplitKDecode.flash_split_k_decode[ctx=128]` | Three shared-address reshards become views; remove their buffers, 33,792 bytes of smem reads/writes and CTA exchange; predicted completion 1403 to 1398 ns |
| `mesh_slice_start.Fixed.scan` | Shared-address reshard becomes a view; remove its buffer and 1024-byte smem reads/writes; subsequent scratch offset 1024 to 0; completion 61 to 58 ns |
| `mesh_slice_start.Strided.scan` | Same structural change; completion 63 to 60 ns |
| `TransposeShard.transpose_shard` | Equal shared physical boxes alias despite device ownership change, as required by the ownership-independent rule; gmem peak 256 to 128 bytes, no redistribution traffic, completion 1 to 0 ns |

The last row is a material estimator limitation: this Reshard decision does not
model inter-device redistribution when the physical boxes agree. The specification
and regression explicitly state that consequence rather than preserving an
inconsistent old traffic assertion.

## TIR text changes against the baseline

All 48 existing successful fixture files change text. Mesh display names remain
those authored in the IR/types. Views are nested on immediate storage partners;
identity views and binding numbers can consequently differ. Seventy-eight static
`For(o_*)` syntax nodes disappear across the corpus. Their expanded issue count,
order and operand addresses are preserved. The two new fixtures have no successful
baseline TIR: one is the new counting witness, the other reproduces the unit-mode
failure.

| Fixture | Removed issue-loop syntax nodes | TensorView syntax count delta | Classification |
|---|---:|---:|---|
| `captured_insert_update` | 0 | +5 | Partner views and mesh names |
| `child_weight` | 0 | +1 | Partner views and mesh names |
| `cta_submesh_thread_scope` | 0 | +3 | Partner views and mesh names |
| `fp8_block_scaled_gemm` | 3 | +27 | Static issues, partner views, mesh names |
| `gemm_8192x17408x5120_register_store` | 3 | +24 | Static issues, partner views, mesh names |
| `gemm_8192x17408x5120_tma_store` | 3 | +28 | Static issues, partner views, mesh names |
| `inplace_output_update` | 0 | +0 | Mesh names |
| `loop_carry_from_param` | 0 | +4 | Partner views and mesh names |
| `loop_carry_rebind` | 0 | +8 | Partner views and mesh names |
| `loop_carry_window` | 0 | +9 | Partner views and mesh names |
| `loop_outputs` | 0 | +11 | Partner views and mesh names |
| `loop_variable_rebound` | 0 | +7 | Partner views and mesh names |
| `mesh_capture_outside` | 0 | +4 | Partner views and mesh names |
| `mesh_views` | 0 | +18 | Partner views and mesh names |
| `mesh_zeros_submesh` | 0 | +1 | Partner views and mesh names |
| `module_instances` | 0 | +11 | Partner views and mesh names |
| `register_arange` | 0 | +1 | Partner views and mesh names |
| `register_full_like` | 0 | +1 | Partner views and mesh names |
| `register_index_window` | 0 | +2 | Partner views and mesh names |
| `register_index_window_offset` | 0 | +2 | Partner views and mesh names |
| `reshard_views` | 0 | +4 | Partner views and mesh names |
| `scalar_binary` | 0 | +1 | Partner views and mesh names |
| `scalar_ceildiv_gate` | 0 | +2 | Partner views and mesh names |
| `sm80_mma_ldmatrix` | 3 | +15 | Static issues, partner views, mesh names |
| `sparse_decode` | 6 | +44 | Static issues, partner views, mesh names |
| `staged_dynamic_inner` | 0 | +6 | Partner views and mesh names |
| `staged_nested` | 0 | +7 | Partner views and mesh names |
| `strided_return_window` | 0 | +5 | Partner views and mesh names |
| `submesh_slice_in_sibling` | 0 | +4 | Partner views and mesh names |
| `wgmma_a_k_major` | 3 | +18 | Static issues, partner views, mesh names |
| `wgmma_a_mn_major` | 3 | +18 | Static issues, partner views, mesh names |
| `wgmma_b_mn_major_grouped_n64` | 3 | -1 | Static issues, partner views, mesh names |
| `wgmma_b_mn_major_wide_lbo` | 3 | +1 | Static issues, partner views, mesh names |
| `wgmma_b_nk` | 3 | -1 | Static issues, partner views, mesh names |
| `wgmma_cast_between_schedules` | 3 | +14 | Static issues, partner views, mesh names |
| `wgmma_cp_async_loads` | 3 | +18 | Static issues, partner views, mesh names |
| `wgmma_cta_grid_4x17` | 3 | +18 | Static issues, partner views, mesh names |
| `wgmma_explicit_windows` | 3 | +21 | Static issues, partner views, mesh names |
| `wgmma_k_slices_of_wide_run` | 3 | +19 | Static issues, partner views, mesh names |
| `wgmma_one_tile_of_larger_output` | 3 | +18 | Static issues, partner views, mesh names |
| `wgmma_repeat_along_k` | 3 | +21 | Static issues, partner views, mesh names |
| `wgmma_repeat_along_n_order` | 3 | +26 | Static issues, partner views, mesh names |
| `wgmma_rs_a_from_accumulator` | 6 | +35 | Static issues, partner views, mesh names |
| `wgmma_rs_a_from_smem` | 3 | +17 | Static issues, partner views, mesh names |
| `wgmma_swizzled_smem` | 3 | +18 | Static issues, partner views, mesh names |
| `wgmma_tma_3stage` | 3 | +18 | Static issues, partner views, mesh names |
| `wgmma_two_schedules` | 6 | +34 | Static issues, partner views, mesh names |
| `zero_trip_insert_slice` | 0 | +3 | Partner views and mesh names |

The `schedule` tutorial contains the same nested-view/static-issue/name changes.
The `showcase` tutorial's only output difference is removal of a redundant reshard
traffic charge under the revised box rule. Tutorial authored code is unchanged.

## Function consolidation and size

`convert_hir_to_tir.py` has 37 function definitions including three local functions,
compared with 97 at the original baseline; `Lowering` has 10 methods versus 68.
Immediately before M6 the counts were 90 and 42. The final names are:

```text
TirBuilder.add
TirBuilder.bind
TirBuilder.build
TirBuilder.region
TirBuilder.tensor
TirBuilder.tensor.build
TirBuilder.value
TirBuilder.transfer
Names.fresh
Names.pair
_label
StoragePlan.regions
StoragePlan.answers
StoragePlan.chain
StoragePlan.identity_refusal
StoragePlan.placement
StoragePlan.runtime_values
StoragePlan.window
StoragePlan.view
StoragePlan.view.addresses
IssuePlan.op
IssuePlan.plan
IssuePlan.source_layout
IssuePlan.points
LoweringContext.dim
Lowering.visit
Lowering.__post_init__
Lowering.run
Lowering.tensor
Lowering.visit_MeshRegion
Lowering.visit_LoopRegion
Lowering.visit_Call
Lowering._lower_call
Lowering.visit_InsertSlice
Lowering.instruction
ConvertHIRToTIR.run
ConvertHIRToTIR.run.lower_module
```

Each M5 function's final destination or deletion basis:

| Before M6 | Final destination | Basis |
|---|---|---|
| `_View.type` | `View.type` | Use stored dataclass fields instead of an accessor or constructor. |
| `_Scope.__init__` | `TirScope` | Use stored dataclass fields instead of an accessor or constructor. |
| `_Scope.add` | `TirBuilder.add` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `_Scope.bind` | `TirBuilder.bind` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `_Scope.build` | `TirBuilder.build` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `_Scope.build.build` | `TirBuilder.build` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `LoweringContext.__init__` | `TirBuilder` | Use stored dataclass fields instead of an accessor or constructor. |
| `LoweringContext.current` | `TirBuilder.blocks[-1]` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `LoweringContext.frame` | `TirScope.mesh` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `LoweringContext.bind` | `TirBuilder.bind` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `LoweringContext.add` | `TirBuilder.add` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `LoweringContext.declaration_scope` | `TirBuilder.region` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `LoweringContext.mesh_scope` | `TirBuilder.region` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `LoweringContext.loop` | `TirBuilder.region` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Names.__init__` | `Names` | Use stored dataclass fields instead of an accessor or constructor. |
| `Names.fresh` | `Names.fresh` | Retain the responsibility. |
| `Names.binding` | `Names.fresh` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `_frame` | `IR/type Mesh` | The IR or operand type already states this information. |
| `_with_frame` | `IR/type ShardLayout.mesh` | The IR or operand type already states this information. |
| `_label` | `_label` | Retain the responsibility. |
| `StorageContext.regions` | `StoragePlan.regions` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.answers` | `StoragePlan.answers` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.chain` | `StoragePlan.chain` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.identity_refusal` | `StoragePlan.identity_refusal` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.contains` | `StoragePlan.answers membership` | Read the immutable answer directly. |
| `StorageContext.offsets` | `StorageAnswer.offsets` | Read the immutable answer directly. |
| `StorageContext.refusal` | `StorageAnswer.refusal` | Read the immutable answer directly. |
| `StorageContext.partner` | `StorageAnswer.partner` | Read the immutable answer directly. |
| `StorageContext.kind` | `StorageAnswer.aliases` | Read the immutable answer directly. |
| `StorageContext.tied` | `StorageAnswer.aliases` | Read the immutable answer directly. |
| `StorageContext.relation` | `StorageAnswer.relation` | Read the immutable answer directly. |
| `StorageContext.staging_region` | `StoragePlan.placement` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.loops` | `StoragePlan.placement` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.register_regions` | `StoragePlan.placement` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.stage` | `TirBuilder.tensor` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.physical_mesh` | `IR/type Mesh` | The IR or operand type already states this information. |
| `StorageContext.runtime_values` | `StoragePlan.runtime_values` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.stage_updates` | `Lowering.visit_LoopRegion` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.tensor` | `TirBuilder.tensor` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.tensor.build` | `TirBuilder.tensor.build` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.allocate` | `TirBuilder.tensor` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.window` | `StoragePlan.window` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.view` | `StoragePlan.view` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `StorageContext.view.addresses` | `StoragePlan.view.addresses` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit` | `Lowering.visit` | Retain the responsibility. |
| `Lowering.__post_init__` | `Lowering.__post_init__` | Retain the responsibility. |
| `Lowering.run` | `Lowering.run` | Retain the responsibility. |
| `Lowering.tensor` | `Lowering.tensor` | Retain the responsibility. |
| `Lowering._known` | `LoweringContext.dim / memo` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._bind_region_args` | `Lowering.visit` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._pair_authored` | `Names.pair` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.dispatch_visit` | `Lowering.visit` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_Constant` | `Lowering.visit` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_Var` | `Lowering.visit` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_Tuple` | `Lowering.visit` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._check_escape` | `TirBuilder.region` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._check_escape.free_vars` | `TirBuilder.region` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_MeshRegion` | `Lowering.visit_MeshRegion` | Retain the responsibility. |
| `Lowering.visit_LoopRegion` | `Lowering.visit_LoopRegion` | Retain the responsibility. |
| `Lowering._dim` | `LoweringContext.dim` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_Call` | `Lowering.visit_Call` | Retain the responsibility. |
| `Lowering._lower_call` | `Lowering._lower_call` | Retain the responsibility. |
| `Lowering._lower_automatic_instruction` | `IssuePlan.op + Lowering.instruction` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_MeshCoord` | `LoweringContext.dim` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_Binary` | `Lowering.visit_Call / instruction` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_TupleGetItem` | `Lowering.visit_Call` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_Slice` | `Lowering.visit_Call` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_Reshard` | `Lowering.visit_Call / IssuePlan.op` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_Transpose` | `IssuePlan.op + Lowering.instruction` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_Empty` | `Lowering.visit_Call` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_Zeros` | `Lowering.visit_Call / TirBuilder.transfer` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_IndexSelect` | `Lowering.visit_Call` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering.visit_InsertSlice` | `Lowering.visit_InsertSlice` | Retain the responsibility. |
| `Lowering.visit_ScheduleOp` | `Lowering.visit_Call` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._lower_instruction` | `Lowering.instruction` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._emit_instruction` | `Lowering.instruction` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._emit_instruction.emit` | `IssuePlan.points` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._shift_fragment` | `StoragePlan.window` | Static issue points and the single storage window derive the offset. |
| `Lowering._instruction_params` | `Projection.params` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._issued_op` | `Lowering.instruction` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._physical_frame` | `IR/type Mesh` | The IR or operand type already states this information. |
| `Lowering._issue_scope` | `TirBuilder.region` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._operand` | `TirBuilder.value` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._whole_operand` | `TirBuilder.value` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._bind_window` | `TirBuilder.value` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._holder_mesh` | `IssuePlan.plan / typed layout` | The IR or operand type already states this information. |
| `Lowering._emit_fill` | `TirBuilder.transfer` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `Lowering._emit_copy` | `TirBuilder.transfer` | Merge into the destination shown; remove the separate wrapper or emitter. |
| `ConvertHIRToTIR.run` | `ConvertHIRToTIR.run` | Retain the responsibility. |
| `ConvertHIRToTIR.run.lower_module` | `ConvertHIRToTIR.run.lower_module` | Retain the responsibility. |

`src/` is net -17 lines against `fdee139e`. It meets the repository's
non-growth rule but misses the plan's -100 target. The revised M3-M6 design adds
explicit plans, immutable leaf answers, lexical builder ownership and independent
static issue projection; retaining their error checks is preferable to compressing
or removing them merely to satisfy the earlier size estimate. The plan is left
unchanged and this exception is reported explicitly.
