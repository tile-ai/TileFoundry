# Step one: describe the published model

Write the published model as authored HIR. One Module per boundary somebody will
implement, its kernels as `@func`s, its weights as `ConstTensor` parameters:

{{fixture: qwen3_5_35b_a3b/model.py:Qwen3_5Router}}

The whole of `qwen3_5_35b_a3b` is written that way. Read the model itself rather
than a description of it: `tilefoundry models qwen3_5_35b_a3b` prints its Module
forest, and `--source` names the directory and lists its files.

## What each part of a declaration means

- A `ConstTensor` parameter is a weight. It lands in `Module.weights`, `load(resource)`
  binds it, and a runtime twin reads it by name; a plain `Tensor` parameter does none
  of that — [core-ir §1](../spec/core-ir.md#1-module).
- `entry` names the function the Module runs by default. What a runtime twin may and
  may not declare is [runtime §1.1](../spec/runtime.md#11-runtimemodule).
- A dimension the model leaves open is a `DimVar` and appears in types. A decode step
  takes the prior cache and returns its own entry; for the fixed-capacity form see
  [hir § CacheUpdate](../spec/hir.md#cacheupdate).
- The root Module declares the target its tree runs on and the topology levels it
  divides over — [target §6](../spec/target.md#6-target-ownership-and-compile-resolution).
- A weight whose published layout differs from the one a `@func` wants is converted by
  `@<func>.converter` — [runtime §1.1.2](../spec/runtime.md#112-weight-converter-and-prepare--forward).
- Dimensions come from the published config. `head_dim` is a published field, not
  `hidden ÷ num_heads`; for this model those differ.
- Variadic tensor operations take one explicit sequence. Write
  `tf.concat([left, right], axis=-1)` or `tf.stack((left, right), axis=0)`, rather
  than passing tensors as separate positional arguments.

## The five access faces

| expression | what you get |
|---|---|
| `Mod.some_func(*all parameters)` | runs it; `ConstTensor` parameters are passed explicitly |
| `Mod.some_child` | the child `Module` node |
| `Mod.lookup("some_func")` | the `ModuleFunction` IR node |
| `loaded.some_func(*activations only)` | runs it; `ConstTensor` parameters come from this loading |
| `loaded.some_child` | the child `LoadedModule` |

## When step one is finished

Prepare real weights first ([runtime §1.1.2](../spec/runtime.md#112-weight-converter-and-prepare--forward)).
Step one is finished when the authored Module agrees with the published
implementation at production dimensions. `check` is the command that says so.
