# TileFoundry CUDA Spec — HIR

## 1. `MatMul` result storage

The target-specific type rule extends the core
[HIR MatMul contract](../../../../../../docs/spec/hir.md#matmul--conv2d--relu--sigmoid--tanh--softmax--layernorm).

- constraints:
  - Under a CUDA Target, a `MatMul` with an operand in `smem` or `rmem` takes
    its result storage from the MMA instructions the Target supports for it:
    each declaration's A, B and C are read in every configuration its patterns
    admit, and a configuration whose A and B admit the operand dtypes and
    storages and whose C admits the result dtype contributes C's storage. One
    storage is the result storage; several are `umat`, for the author to
    resolve; none MUST be refused, naming the Target, both operands and the
    result dtype. Operands in `gmem` or `umat` keep the target-neutral rule,
    and so does every other Target.
