# Issue 237 example migration

Generated schedules follow the four tools/gen_scheduled templates. Fragment permutations below preserve every (extent, stride, mesh axis) triple. Output initialization moves into the owning thread mesh; no extra copies or pipeline-depth changes are introduced. Model and checkpoint paths use ModuleList and layers.{i}.

## Generator layout log (D35)

- `tools/gen_scheduled_attention.py:170:frag`: `((8 @ {c}.lane8, 2, 4 @ {c}.warp, 2, 4 @ {c}.lane4, {n // 8}), (1, 8, 16, 64, 128, 512))` → `((4 @ {c}.warp, 2, 8 @ {c}.lane8, {n // 8}, 4 @ {c}.lane4, 2), (16, 8, 1, 512, 128, 64))`.
- `tools/gen_scheduled_attention.py:175:pfrag`: `((4, 8 @ {c}.lane8, 2, 4 @ {c}.warp, 2, 4 @ {c}.lane4, 2), (1024, 1, 8, 16, 64, 128, 512))` → `((4 @ {c}.warp, 2, 8 @ {c}.lane8, 4, 2, 4 @ {c}.lane4, 2), (16, 8, 1, 1024, 512, 128, 64))`.
- `tools/gen_scheduled_gdn.py:223:frag`: `((8 @ {c}.lane8, 2, 4 @ {c}.warp, 2, 4 @ {c}.lane4, {n // 8}), (1, 8, 16, 64, 128, 512))` → `((4 @ {c}.warp, 2, 8 @ {c}.lane8, {n // 8}, 4 @ {c}.lane4, 2), (16, 8, 1, 512, 128, 64))`.
- `tools/gen_scheduled_gdn.py:229:frag_n2`: `((2 @ {c}.group, 8 @ {c}.lane8, 2, 4 @ {c}.warp, 2, 4 @ {c}.lane4, {h // 8}), ({64 * h}, 1, 8, 16, 64, 128, 512))` → `((4 @ {c}.warp, 2, 8 @ {c}.lane8, 2 @ {c}.group, {h // 8}, 4 @ {c}.lane4, 2), (16, 8, 1, {64 * h}, 512, 128, 64))`.
- `tools/gen_scheduled_gdn.py:234:rowv_n2`: `((1, 8 @ {c}.lane8, 2, 4 @ {c}.warp, 1, 1, 1), (0, 1, 8, 16, 0, 0, 0))` → `((4 @ {c}.warp, 2, 8 @ {c}.lane8, 1, 1, 1, 1), (16, 8, 1, 0, 0, 0, 0))`.
- `tools/gen_scheduled_gdn.py:240:colv_n2`: `((2 @ {c}.group, 1, 1, 1, 2, 4 @ {c}.lane4, {h // 8}), ({64 * h}, 0, 0, 0, 64, 128, 512))` → `((1, 2 @ {c}.group, {h // 8}, 4 @ {c}.lane4, 2, 1, 1), (0, {64 * h}, 512, 128, 64, 0, 0))`.
- `tools/gen_scheduled_gdn.py:562:afrag`: `(({ktiles}, 8 @ {c}.lane8, 2, 4 @ {c}.warp, 2, 4 @ {c}.lane4, 2), (1024, 1, 8, 16, 64, 128, 512))` → `((4 @ {c}.warp, 2, 8 @ {c}.lane8, {ktiles}, 2, 4 @ {c}.lane4, 2), (16, 8, 1, 1024, 512, 128, 64))`.
- `tools/gen_scheduled_gdn.py:567:colv1`: `((1, 1, 1, 2, 4 @ {c}.lane4, {n // 8}), (0, 0, 0, 64, 128, 512))` → `((1, {n // 8}, 4 @ {c}.lane4, 2, 1, 1), (0, 512, 128, 64, 0, 0))`.
- `tools/gen_scheduled_gdn.py:576:frag_2n`: `((2, 8 @ {c}.lane8, 2, 4 @ {c}.warp, 2, 4 @ {c}.lane4, {n // 8}), ({64 * n}, 1, 8, 16, 64, 128, 512))` → `((4 @ {c}.warp, 2, 8 @ {c}.lane8, 2, {n // 8}, 4 @ {c}.lane4, 2), (16, 8, 1, {64 * n}, 512, 128, 64))`.
- `tools/gen_scheduled_gdn.py:463:UT`: `((2, 2 @ cw.group, 8 @ cw.lane8, 2, 4 @ cw.warp, 2, 4 @ cw.lane4, 4), (2048, 4096, 1, 8, 16, 64, 128, 512))` → `((2, 4 @ cw.warp, 2, 8 @ cw.lane8, 2 @ cw.group, 4, 4 @ cw.lane4, 2), (2048, 16, 8, 1, 4096, 512, 128, 64))`.
- `tools/gen_scheduled.py:78:acc_layout`: `((2 @ _compute.group, 8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, {BN // 8}), ({BN * 64}, 1, 8, 16, 64, 128, 512))` → `((2 @ _compute.group, 4 @ _compute.warp, 2, 8 @ _compute.lane8, {BN // 8}, 4 @ _compute.lane4, 2), ({BN * 64}, 16, 8, 1, 512, 128, 64))`.

## Plain Bitcast inventory (D69)

All 85 affected instances at 53 source sites match the program intent under row-major correspondence: 66 transposes and 19 identities. The independent NumPy mapping comparison found no compensating colex layouts, so all authored Bitcast layouts are retained.
