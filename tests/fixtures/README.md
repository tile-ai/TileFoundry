# tests/fixtures

What is here and how to reach it.

## `logical/` and `placed/` — complete reference programs

Whole programs. `logical/` leaves placement to scheduling; `placed/` states its
own meshes and topologies.

```python
from tests.fixtures.logical.hir_composition import Expert
from tests.fixtures.placed.rmsnorm import RmsnormModule
```

## `shapes/` — small structures more than one test module reads

| file | what it holds | how it is reached |
|---|---|---|
| `tile_window_add.py` | `tile_window_add`, a `(10,4)` window stepped by 4 | `extract`, `evaluate`, `HirToTirPass` |
| `moved_tile_window_add.py` | `moved_tile_window_add`, the same window moved by 6 | `extract`, `HirToTirPass` |
| `gemm_rms_norm.py` | `gemm_rms_norm`, f32 `(2,4)x(4,2)`, no target | `extract`, `candidate_atoms` |
| `bf16_gemm_rms_norm.py` | `bf16_gemm_rms_norm`, bf16 `(64,128)x(128,64)` on `nvidia.h200_sxm` | `schedule`, pipeline solving |
| `scaled_child.py` | `ScaledChild`, a child Module holding weight `w` | bound as a child, addressed as `<binding>.w` |
| `paired_scaled_parent.py` | `PairedScaledParent`, entry `both(w)`, binds `ScaledChild` as `left` and `right` | `check_program`, `load` |
| `fused_scaled_parent.py` | `FusedScaledParent`, entry `fused(x)`, binds `ScaledChild` as `scaled` | `load`, `prepare` |
| `composed_leaf_source.py` | `composed_leaf_source(dim_name)`, the composed root as DSL text | written to a file for the CLI to read |

```python
from tests.fixtures.shapes.gemm_rms_norm import gemm_rms_norm
from tests.fixtures.shapes.scaled_child import ScaledChild
from tests.fixtures.shapes.composed_leaf_source import composed_leaf_source

source.write_text(composed_leaf_source("n_cli"), encoding="utf-8")
```

Importing `paired_scaled_parent` or `fused_scaled_parent` brings `ScaledChild`
with it.

## `tests/models/` — the registered real models

The model catalog `tilefoundry models` reads. The integration and installed
suites run them.
