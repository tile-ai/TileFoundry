# Issue 237 example migration

The migration keeps tile shapes, atoms, mesh ownership, ring depths and copy instructions. Long schedules use explicit destination operands at 309 sites (including 13 TMA stores). Short schedules are generated from the four tools/gen_scheduled templates; ModuleList paths use layers.{i}. Carry and return authoring stays unchanged; dynamic staged rings use the compiler counter (D68).

## Long fragment layouts (D35/D45)

Every row preserves the (extent, stride, mesh axis) multiset. Line numbers refer to the migrated source.

- `scheduled/attention.py:58`: `((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 1), (64, 1, 8, 16, 0))` → `((2 @ c.group, 4 @ c.warp, 2, 8 @ c.lane8, 1), (64, 16, 8, 1, 0))`.
- `scheduled/attention.py:59`: `((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 1), (64, 1, 8, 16, 0))` → `((2 @ c.group, 4 @ c.warp, 2, 8 @ c.lane8, 1), (64, 16, 8, 1, 0))`.
- `scheduled/attention.py:60`: `((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 1), (64, 1, 8, 16, 0))` → `((2 @ c.group, 4 @ c.warp, 2, 8 @ c.lane8, 1), (64, 16, 8, 1, 0))`.
- `scheduled/attention.py:73`: `((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 1), (64, 1, 8, 16, 0))` → `((2 @ c.group, 4 @ c.warp, 2, 8 @ c.lane8, 1), (64, 16, 8, 1, 0))`.
- `scheduled/attention.py:76`: `((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 1), (64, 1, 8, 16, 0))` → `((2 @ c.group, 4 @ c.warp, 2, 8 @ c.lane8, 1), (64, 16, 8, 1, 0))`.
- `scheduled/attention.py:78`: `((2 @ c.group, 5, 4 @ c.warp, 2, 8 @ c.lane8, 2, 4 @ c.lane4, 2), (5120, 1024, 16, 8, 1, 512, 128, 64))` → `((2 @ c.group, 4 @ c.warp, 2, 8 @ c.lane8, 5, 2, 4 @ c.lane4, 2), (5120, 16, 8, 1, 1024, 512, 128, 64))`.
- `scheduled/gdn.py:107`: `((8 @ s.lane8, 2, 4 @ s.warp, 1), (1, 8, 16, 0))` → `((4 @ s.warp, 2, 8 @ s.lane8, 1), (16, 8, 1, 0))`.
- `scheduled/gdn.py:111`: `((8 @ s.lane8, 2, 4 @ s.warp, 1), (1, 8, 16, 0))` → `((4 @ s.warp, 2, 8 @ s.lane8, 1), (16, 8, 1, 0))`.
- `scheduled/gdn.py:113`: `((8 @ s.lane8, 2, 4 @ s.warp, 1), (1, 8, 16, 0))` → `((4 @ s.warp, 2, 8 @ s.lane8, 1), (16, 8, 1, 0))`.
- `scheduled/gdn.py:234`: `((8 @ s2.lane8, 2, 4 @ s2.warp, 1), (1, 8, 16, 0))` → `((4 @ s2.warp, 2, 8 @ s2.lane8, 1), (16, 8, 1, 0))`.
- `scheduled/gdn.py:335`: `((8 @ s0.lane8, 2, 4 @ s0.warp, 1), (1, 8, 16, 0))` → `((4 @ s0.warp, 2, 8 @ s0.lane8, 1), (16, 8, 1, 0))`.
- `scheduled/gdn.py:343`: `((8 @ s1.lane8, 2, 4 @ s1.warp, 1), (1, 8, 16, 0))` → `((4 @ s1.warp, 2, 8 @ s1.lane8, 1), (16, 8, 1, 0))`.

## Targeted finalize retry

Only rows that previously failed were retried here. Full artifact regeneration, normalized group comparisons and address audits run in the final gate (D67).

| Selector/file | Finalize |
|---|---|
| `scheduled/gdn.py:DeltaRuleIntra1024.intra_wup` | PASS |
| `scheduled/gdn.py:DeltaRuleIntra4096.intra_wup` | PASS |
| `scheduled/gdn.py:DeltaRuleIntra8192.intra_wup` | PASS |
| `scheduled/gemm.py:Down1024.pieces` | PASS |
| `scheduled/gemm.py:Down1024.whole` | PASS |
| `scheduled/gemm.py:Down4096.whole` | PASS |
| `scheduled/gemm.py:Down8192.whole` | PASS |
| `scheduled/gemm.py:DownTail1024.tail` | PASS |
| `scheduled/gemm.py:GateUp1024.whole` | PASS |
| `scheduled/gemm.py:GateUp4096.whole` | PASS |
| `scheduled/gemm.py:GateUp8192.whole` | PASS |
| `scheduled/gemm.py:GateUpTail1024.tail` | PASS |
| `scheduled/gemm.py:InProj1024.whole` | PASS |
| `scheduled/gemm.py:InProj4096.whole` | PASS |
| `scheduled/gemm.py:InProj8192.whole` | PASS |
| `scheduled/gemm.py:OProj1024.whole` | PASS |
| `scheduled/gemm.py:OProj4096.whole` | PASS |
| `scheduled/gemm.py:OProj8192.whole` | PASS |
| `scheduled/gemm.py:OProjTail1024.tail` | PASS |
| `scheduled/gemm.py:OutProj1024.whole` | PASS |
| `scheduled/gemm.py:OutProj4096.whole` | PASS |
| `scheduled/gemm.py:OutProj8192.whole` | PASS |
| `scheduled/gemm.py:OutProjTail1024.tail` | PASS |
| `scheduled/gemm.py:QKV1024.pieces` | PASS |
| `scheduled/gemm.py:QKV1024.whole` | PASS |
| `scheduled/gemm.py:QKV1024.whole_last` | PASS |
| `scheduled/gemm.py:QKV8192.whole_b` | PASS |
| `scheduled/gemm.py:QKV8192.whole_last` | PASS |
| `scheduled/gemm.py:QKVTail1024.tail` | PASS |
| `scheduled/gemm.py:QKVTail8192.tail` | PASS |
