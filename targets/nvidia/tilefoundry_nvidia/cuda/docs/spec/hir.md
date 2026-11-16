# TileFoundry CUDA Spec — HIR

## 1. `MatMul` result storage

The target-specific type rule extends the core
[HIR MatMul contract](../../../../../../docs/spec/hir.md#matmul--conv2d--relu--sigmoid--tanh--softmax--layernorm).

- constraints:
  - Under a CUDA Target, a `MatMul` with an operand in `smem` or `rmem` takes
    its result storage from MMA instructions. The Target supports these
    instructions for the op.
    Each declaration's A, B and C are read in every configuration its patterns
    admit.
    A configuration contributes C's storage when A and B admit the operand
    dtypes and storages and C admits the result dtype.
    If the configurations contribute one storage, that storage is the result
    storage.
    If they contribute several storages, the result storage is `umat`.
    The author resolves it.
    If they contribute none, the Target MUST refuse the op.
    The refusal MUST name the Target, both operands and the result dtype.
    Operands in `gmem` or `umat` keep the target-neutral rule.
    Every other Target keeps the target-neutral rule.
