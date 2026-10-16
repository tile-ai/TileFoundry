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
