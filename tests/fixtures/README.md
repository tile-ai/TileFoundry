# tests/fixtures

What is here and how to reach it.

## `logical/` and `placed/` — complete reference programs

Whole programs. `logical/` leaves placement to scheduling; `placed/` states its
own meshes and topologies.

```python
from tests.fixtures.logical.hir_composition import Expert
from tests.fixtures.logical.module_context import ContextTree
from tests.fixtures.placed.rmsnorm import RmsnormModule
```

## `shapes/` — small structures more than one test module reads

| file | what it holds | how it is reached |
|---|---|---|
| `window_programs.py` | fixed-tail and compile-time-moved tile windows | `extract`, `evaluate`, `HirToTirPass` |
| `matmul_programs.py` | static/dynamic CUDA and AMX matmuls plus rms-norm compositions | `extract`, target facts, scheduling |
| `scaled_modules.py` | one weighted child plus single- and double-binding parents | `check_program`, `load`, `prepare` |
| `composed_leaf_source.py` | `composed_leaf_source(dim_name)`, the composed root as DSL text | written to a file for the CLI to read |

```python
from tests.fixtures.shapes.matmul_programs import gemm_rms_norm
from tests.fixtures.shapes.scaled_modules import ScaledChild
from tests.fixtures.shapes.composed_leaf_source import composed_leaf_source

source.write_text(composed_leaf_source("n_cli"), encoding="utf-8")
```

## `tests/models/` — the registered real models

The model catalog `tilefoundry models` reads. The integration and installed
suites run them.
