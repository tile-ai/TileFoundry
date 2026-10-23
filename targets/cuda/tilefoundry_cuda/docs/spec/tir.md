# TileFoundry CUDA Spec — TIR

CUDA-owned instructions and the device lowering of shared TIR operations.

## Sync lowering

### Supported slices and the barrier mapping

The participant set MUST be a single contiguous thread interval `[base,
base+count)`. Verify MUST reject (never broaden or split):

- a non-contiguous slice (e.g. a lane subset spanning warps);
- a cross-warp range that is not warp-aligned (`base` and `count` not both
  multiples of 32);
- a dynamic / inconsistent / unsupported-topology mesh.

A valid participant set maps to exactly one hardware barrier:

| participant set | barrier |
|---|---|
| whole block, more than one warp (`base==0`, `count==domain`) | `__syncthreads()` |
| whole block that is one warp | `__syncwarp()` |
| a contiguous lane subset within one warp | `__syncwarp(mask)` under a participant predicate |
| a warp-aligned contiguous multi-warp subset | a named `bar.sync <id>, <count>` under a participant predicate |
| the full mesh over the `cta` topology (all CTAs of the grid) | the grid-wide software barrier ([runtime §2.6](./runtime.md#26-cudaops)) |

Codegen MUST guard the `__syncwarp(mask)` and `bar.sync` cases with the
participant predicate `base <= tid < base+count` (`tid =
program_id<thread>()`): a non-participant thread MUST NOT execute the barrier,
and every participant MUST execute the same id and count.

The first four rows synchronize threads **within one block**; their participant
set is the contiguous thread interval above. A mesh whose topologies are all the
`cta` topology instead synchronizes **CTAs across the grid** — `program_id<cta>`
ranges over the launch's blocks — and maps to the grid-wide software barrier.
Only the **full** cta mesh participates: a cta slice (a subset of CTAs) has no
supported barrier and MUST be rejected at verify. The grid barrier's correctness
requires every CTA of the launch to be co-resident; that co-residency is the
launch's occupancy contract, not something the barrier can enforce. The
grid-barrier device helper and its counter protocol are specified in
[runtime §2.6](./runtime.md#26-cudaops).

### Named-barrier id allocation

A sub-CTA `bar.sync` MUST carry a named-barrier id, allocated implicitly during
codegen, per kernel. Id `0` is reserved for the whole-CTA barrier; sub-CTA syncs
draw ids from `1..15`. Each emitted `bar.sync` MUST take the next free id; a
sync op node emits once, so a loop body reuses its id. A kernel requiring more
than 15 distinct named barriers MUST error; an id MUST NOT be reused across
distinct sync sites.

## Abort lowering

The device emitter renders `Abort` as `__trap();`; host contexts use
`assert(false);`, so a runtime hit is loud rather than silent.

## TiledMma
```python
class TiledMma(Op):
    """Effect form; matrix-multiply-accumulate ``acc += lhs @ rhs``.

    Attributes:
        acc: input; accumulator fragment.
        lhs: input; left-hand operand fragment.
        rhs: input; right-hand operand fragment.
        atom: attribute; required compile-time ``MmaAtom`` declaration.
        execution_mesh: attribute; optional warp-aligned execution mesh.
    """

    acc: Tensor
    lhs: Tensor
    rhs: Tensor
    atom: MmaAtom
    execution_mesh: Mesh | None = None
```
- constraints:
  - matrix-multiply-accumulate `acc += lhs @ rhs`; target lowering follows the atom calling convention
    [below](#declarative-mma-atoms-and-ttiled_mma).
  - `acc` declares `READ | WRITE`; `lhs` and `rhs` declare `READ`.

## Tensor-op lowering

### Reduce

`workspace` is smem: each warp posts a partial and reads the others back.
It is present only for cross-warp staging, has rank one, and holds at least
`ceil(threads / 32)` slots over the source mesh's static positive extents.
Its access relation reaches those slots from every iteration; the runtime
chooses which warp uses each slot. All forms lower to
`tilefoundry::ops::reduce<Op, Axes>(src, dst[, workspace])`. Sharded extents
and tiers are runtime choices, not TIR dispatch parameters.

### Dot

Without a workspace the contraction is within a warp; with one, every warp
posts a partial into shared memory. A workspace requires a `ShardLayout` on
`lhs`: the number of warps and the barrier come from that mesh. Both forms
lower to `tilefoundry::ops::dot(lhs, rhs, dst[, workspace])`.

### Elementwise coverage

`Where` and `Arange` have no device emitter in this target.

## Launch attributes

`CudaLaunchAttr` is owned by `tilefoundry_cuda.ir.tir.launch`:

```python
class CudaLaunchAttr(IntEnum):
    """Select a CUDA launch attribute carried by shared LaunchAttrs."""
    ...
```

`LaunchAttrs.entries` carries these selector/value pairs. CUDA lowering MUST
reject unsupported selectors or values, including unsupported cluster and stream
configuration. Launch geometry is derived in codegen and emitted into the host
entry; it is not an extra runtime metadata field.

## Declarative MMA atoms and `T.tiled_mma`

An MMA instruction is a target-owned `MmaAtom` declaration. The declaration
class states its authored parameters, required physical execution mesh, target
capability, and the `TensorPattern` read for each `A`, `B`, and `C` role. An
instance binds the parameters for one call; it does not carry a second copy of
concrete fragment layouts that could drift from those patterns.

```python
class MmaAtom(Printable):
    namespace: str
    execution_mesh: Mesh
    capability: str
    A: TensorPattern | SwitchPattern
    B: TensorPattern | SwitchPattern
    C: TensorPattern | SwitchPattern
    parameters: tuple[ParamDef, ...]
    bindings: dict[str, object]
    mesh: Mesh | None

    def role(self, role: str) -> TensorPattern: ...
    def execution_mesh_pattern(self) -> MeshPattern: ...
    def print(self, printer: PrinterBase, ctx=None) -> str: ...

def variants(
    parameters: tuple[ParamDef, ...],
    *,
    vary_defaulted: bool,
    values: Callable[[ParamDef], tuple] | None = None,
) -> tuple[dict, ...]: ...  # ir/pattern/utils.py
```

- constraints:
  - `parameters` MUST preserve declaration order. Construction MUST reject an
    unknown binding and a value refused by its `ParamDef.pattern`; an omitted
    parameter MUST take the value implied by earlier bindings or its declared
    default, and otherwise construction MUST fail. `bindings` then holds every
    parameter, in declaration order.
  - A string bound to a `DType` parameter MUST resolve through
    `DType.from_name` at construction, so `dtype="bf16"` and `DType.bf16` bind
    the same value; whether that dtype is one the declaration takes is still its
    `ParamDef.pattern`'s answer. An Enum parameter takes only its Enum members;
    no string spelling of a member is accepted.
  - `variants(declaration.parameters, vary_defaulted=..., values=...)` MUST
    return every binding of the parameters, in declaration order, that their
    patterns admit under the earlier bindings. It enumerates bindings; it does
    not build atoms or infer parameters from operand types. Enum parameters
    take their members in declaration order and `DType` parameters every
    declared dtype; `values` supplies the others, which stay unbound without it
    or when it answers `None`, while an empty answer admits no binding. A
    parameter with a default is varied only when `vary_defaulted` is true. No
    parameters give one empty binding. `schedule candidates` calls it with
    `vary_defaulted=False` and binds `int` parameters to `1..` the site's
    largest extent; CUDA `MatMul` typing calls it with `vary_defaulted=True`
    and leaves `int` parameters unbound.
  - Role-pattern modes MUST follow the logical row-major correspondence in
    [semantic-analysis §3.1](../../../../../docs/spec/semantic-analysis.md#31-logical-shape-to-layout-domain):
    A `(M, K)`, B `(K, N)`, and C `(M, N)`. Split modes MUST place each element
    on the thread specified by PTX; strides retain their physical values.
    Repeat modes belong to their logical axes, and operand tiles remove those
    axes' outer digits rather than a shared prefix of the entire layout.
  - `MmaAtom` is `Printable` ([core-ir §2.4](../../../../../docs/spec/core-ir.md#24-printable-and-printerbase)):
    `print(printer, ctx)` is the atom's importable DSL source. It states every
    binding in declaration order, implied ones included, writes an Enum as
    `<namespace>.<Enum>.<member>`, and hands every other value, the mesh
    included, to `printer.print(value, ctx)`, so a `DType` prints as its name
    and imports land in the caller's context. The atom never builds a printer.
    A value with no Python form fails explicitly. `repr(atom)` is a local
    diagnostic with the same bindings and is not an import surface.
  - `role("A")`, `role("B")`, and `role("C")` MUST resolve the declaration's
    role pattern under the instance bindings. The logical TIR orientation is
    always A `(M,K)`, B `(K,N)`, C `(M,N)`; each role pattern separately states
    the fragment's physical arrangement.
  - The registered `T.tiled_mma` access relation is the sole source of the map
    from each operand axis to the operation's work axes. Atom declarations state
    operand shapes and arrangements, but MUST NOT repeat that axis map.
  - `execution_mesh_pattern()` MUST require the declaration's exact participant
    count at an aligned offset. `mesh`, when present, binds the atom to one
    concrete frame and MUST match the active frame at verify.
  - `capability` MUST use the emitted instruction-family name and be present in
    the active CUDA architecture's `capabilities`. `resource` MUST name the
    execution engine with an `_engine` suffix. These are separate axes: support
    for an instruction family does not itself state an engine rate.

The public declarations are `T.cuda.sm80.Mma()` (BF16 `16x8x16`, F32
accumulator, register A/B/C over one warp), `T.cuda.sm89.Mma()` (E4M3
`16x8x32`, F32 accumulator laid out as the SM80 one, register A/B/C over one
warp) and
`T.cuda.sm90.Wgmma(n=..., dtype=..., form=..., a_major=..., b_major=..., mesh=...)`
(`dtype` is `bf16`, `f16` or `fp8e4m3`; `64 x n x k` over one warpgroup with
an F32 accumulator, where `k` is 32 bytes of `dtype`). `dtype` has no default.
`a_major` and `b_major` default to `Major.MN`. A 16-bit B may be either major
and a 16-bit A read from registers is K-major; an `fp8e4m3` A and B are
K-major only, so that `dtype` implies both and an explicit `Major.MN` is
refused. `Form` and `Major` live beside `Wgmma` under `T.cuda.sm90`.

### Calling convention

Load, compute, and store are separate effect statements under an enclosing
`MeshScope` ([§1.2](../../../../../docs/spec/tir.md#12-structural-stmts-tirstmts)). `T.tiled_mma` has exactly
three input operands plus one required compile-time `atom` attribute:

```python
# example
T.tiled_mma(acc, lhs, rhs, atom=T.cuda.sm80.Mma())
```

- `acc` MUST match `atom.role("C")` and is read-write.
- `lhs` MUST match `atom.role("A")` and is read-only.
- `rhs` MUST match `atom.role("B")` and is read-only.
- A, B, and C shapes MUST be `(M,K)`, `(K,N)`, and `(M,N)` respectively;
  operands MUST use the atom's declared dtypes and fragment arrangements.
- `TiledMma` generates its execution-mesh declaration from the atom's lane
  shape and strides, preceded by zero or more group modes captured by
  `StarPattern`. The whole matched layout MUST be forward and injective,
  and its offset MUST be aligned to the atom's participant count. Operand
  meshes retain the atom's exact lane axes.
- The active physical mesh MUST satisfy `atom.execution_mesh_pattern()`. If the atom
  carries `mesh=...`, its affine offset and ordered lanes MUST equal the active
  frame.

There is one TIR MMA op: `T.tiled_mma`. There is no optional-atom or bare-MMA
path. Per-target emission dispatches on the atom ([target](./target.md)); the
current CUDA emitter accepts the SM80 and SM89 declarations and rejects WGMMA
because this stage provides no WGMMA runtime emitter.

## Async copy lowering

Shared `CopyAsync`, `CpAsyncCommit`, and `CpAsyncWait` are defined in
[core TIR](../../../../../docs/spec/tir.md#async-copy-ops-tirasync).
`CopyAsync` requires the `cp.async` capability and lowers to
`tilefoundry::ops::copy_async(src, dst)`. An indexed valid row uses `cp.async`;
an invalid row with zero fill uses its zero-fill source-size form, while
nonzero fill uses `st.shared` with the destination dtype's repeated pattern.

### CopyAsyncBulk

A staging copy whose completion lands on an mbarrier, and not a tier of
`CopyAsync`: there every thread issues its own load and a commit closes the
group, so the thread that issues is the thread that waits; here a consumer can
wait for a tile it did not fetch.

Which instruction carries it is the runtime's choice from the operand shard
layouts, not something this op names: a contiguous run takes `cp.async.bulk`,
anything else takes an element path ([runtime
§2.6](./runtime.md#26-cudaops)).
Carrying that on the op would be codegen selecting a tier, which
[§2.3](../../../../../docs/spec/tir.md#23-tir-ops) forbids.

The tensor forms (`cp.async.bulk.tensor.Nd`) take a host-encoded `TensorMap` in
place of a size, which is a different operand list rather than a different tier,
and are outside this op.

```python
class CopyAsyncBulk(Op):
    """Effect form; gmem→smem staging copy completing on an mbarrier.

    Attributes:
        src: input; gmem source tile.
        dst: input; smem destination tile.
        barrier: input; smem mbarrier the completion lands on.
    """

    src: Tensor
    dst: Tensor
    barrier: Tensor
```
- constraints:
  - `src` is gmem, `dst` is smem, `barrier` is smem.
  - `src` and `dst` agree in dtype and shape; the copy moves bytes and does not
    convert them.
  - Nothing blocks: the copy may still be in flight when the issuing thread
    reaches the next statement.
  - Consumers wait with `MBarrierWaitParity`. The arrival that declares the
    transferred bytes is the implementation's, issued on the same instruction as
    the copy; a caller pairing this with its own `MBarrierArriveExpectTx` would
    be declaring a count the op already knows.
  - It requires the target's `cp.async.bulk` capability.
  - Lowers to `tilefoundry::ops::copy_async_bulk(src, dst, bar)`
    ([runtime §2.6](./runtime.md#26-cudaops)). `barrier` is a tensor here
    because that is what TIR names a piece of shared memory with, and a word to
    the runtime, so the emitted call hands over the word's own address.

### CopyAsyncTensor

`T.copy_async_tensor` declares the SM90 tensor-map form separately from
`CopyAsyncBulk`: its operands describe a tensor-map global layout and the box
landed in shared memory, rather than carrying an explicit mbarrier operand.

```python
class CopyAsyncTensor(Op):
    src: Tensor
    dst: Tensor
    execution_mesh: Mesh | None = None
```

- constraints:
  - Exactly one end is gmem and one is smem; dtype and logical shape agree.
  - The global end is a static tensor-map layout of at most five dimensions,
    with one contiguous mode and every other byte stride a multiple of 16.
    The shared end is an at-most-five-dimensional box, each extent at most
    256, optionally using a 32-, 64-, or 128-byte TMA swizzle.
  - Both ends walk the same tile modes. The execution mesh is one aligned warp,
    preceded by zero or more size-one modes with arbitrary strides. The whole
    matched layout MUST be forward and injective; its offset is a multiple of 32.
  - The declaration requires the target's `cp.async.bulk.tensor` capability.
    CUDA codegen MUST reject it until host-encoded tensor-map construction
    exists; this stage does not silently lower it to another copy instruction.

### CopyReduceAsyncTensor

```python
class CopyReduceAsyncTensor(Op):
    src: Tensor
    dst: Tensor
    kind: str = "add"
    execution_mesh: Mesh | None = None
```

- constraints:
  - The source is a shared-memory TMA box and MAY carry `Partial` state;
    the destination is an explicit global tensor-map operand, read and written.
    Shape, dtype and contiguous tile modes MUST agree. Ordinary tensor-map
    copies continue to reject `Partial` state.
  - Only `kind="add"` is supported. The logical result is `dst + src`;
    the evaluator's source already represents the sum of its parts.
  - One aligned warp participates, with the same execution-mesh declaration as
    `CopyAsyncTensor`. The capability is `cp.async.bulk.tensor` and the resource
    is `tma_engine`. Partial state MUST NOT select instruction owner levels;
    owner selection treats Partial as Broadcast while retaining Split placement.
  - Cost MUST charge one source read and one destination write per unit, no
    destination read and no FLOPs. The hardware addition occurs in L2; analysis
    reports the write as gmem traffic under the existing leaf-storage rule.
  - CUDA codegen MUST reject it until host-encoded tensor-map support exists.

### LdMatrix

```python
class LdMatrix(Op):
    src: Tensor
    dst: Tensor
    execution_mesh: Mesh | None = None
```

- constraints:
  - `src` is a shared-memory `(16, 16)` bf16 tile and `dst` is exactly the
    register A fragment declared by `T.cuda.sm80.Mma()`; their dtype and shape
    agree through explicit between-operand constraints.
  - One canonical SM80 warp issues the operation. It requires the target's
    `ldmatrix` capability and lowers to `tilefoundry::ops::ldmatrix`.
  - The source and destination TV layouts follow CuTe `SM75_U32x4_LDSM_N`.

### StMatrix

```python
class StMatrix(Op):
    src: Tensor
    dst: Tensor
    execution_mesh: Mesh | None = None
```

- constraints:
  - `src` is a bf16 or f16 `(16, 16)` register fragment with the per-thread
    ownership of CuTe `SM90_U32x4_STSM_N`: four 32-bit registers per thread.
  - `dst` is shared memory with whole contiguous 16-byte vectors. Source and
    destination shape and dtype MUST agree.
  - One warp (32 threads) issues the CuTe `SM90_U32x4_STSM_N` transfer
    of a 16-by-16 b16 tile. Larger scheduled operands are projected into warp
    issues. The capability is `stmatrix` and the resource is `tensor_core_engine`.
  - CUDA codegen MUST reject it until stmatrix runtime support exists.

## Barrier object Ops (`tir.sync.mbarrier_*`)

A Hopper mbarrier is a 64-bit shared-memory word carrying an arrival count, a
transaction-byte count and a phase parity. It is not `Sync`
([§1.5](../../../../../docs/spec/tir.md#15-sync)): `Sync` is a whole-mesh rendezvous every participant reaches,
while these let a producer signal completion of work the consumer did not
perform — which is what an asynchronous copy needs, since the thread that issues
one is not the thread that waits for it.

**Each lowers to its instruction, not to a runtime entry, and the runtime
publishes no `ops::` entry for any of them:** an mbarrier is a shared-memory
word, so nothing here reads a `ShardLayout` and none of it is an op
([runtime §2.6](./runtime.md#26-cudaops)). Each entry below names the
`mbarrier.*` instruction its emitter writes at the call site, together with the
generic-to-shared conversion the instruction takes — they name `.shared::cta`
explicitly rather than leaving the assembler to redo that window conversion on
every use.

The group is what a `CopyAsyncBulk` ring needs and no more: arm the word, arrive on it
declaring bytes, wait on its phase, release it. A bare `mbarrier.arrive` is
absent because `ops::copy_async_bulk` issues its own for the strided tier, and a bare
`mbarrier.expect_tx` because nothing pairs with it.

### MBarrierInit

```python
class MBarrierInit(Op):
    """Effect form; arm a barrier for a fixed number of arrivals.

    Attributes:
        barrier: input; smem barrier object.
        arrive_count: attribute; arrivals that complete one phase.
    """

    barrier: Tensor
    arrive_count: int
```
- constraints:
  - `barrier` is smem; the instructions take a shared-window address.
  - `arrive_count` is a positive compile-time count. A phase needing zero
    arrivals is complete before anything is produced, which makes every
    consumer's wait a no-op.
  - One thread initialises, and a `Sync` covering every thread that will use the
    barrier separates this from the first arrival or wait.
  - Lowers to `mbarrier.init.shared::cta.b64`, written at the call site with
    `arrive_count` as an inline operand: the runtime publishes no entry for it
    ([runtime §2.6](./runtime.md#26-cudaops)).

### MBarrierArriveExpectTx

```python
class MBarrierArriveExpectTx(Op):
    """Effect form; arrive and declare asynchronous bytes in one instruction.

    Attributes:
        barrier: input; smem barrier object.
        tx_bytes: attribute; bytes the paired copy delivers to this phase.
    """

    barrier: Tensor
    tx_bytes: int
```
- constraints:
  - `barrier` is smem and `tx_bytes` is a positive compile-time count.
  - The phase completes when both the arrivals and the byte count are satisfied,
    so one wait covers a copy the waiting thread did not issue.
  - `tx_bytes` MUST equal the bytes the paired copy delivers. A phase expecting a
    different count never completes, and that failure presents as a hang rather
    than as a wrong value.
  - This is not paired with a `CopyAsyncBulk`, which declares its own bytes on the
    instruction that issues the copy. It belongs to a producer issuing one
    itself.
  - It requires the target's `mbarrier.expect_tx` capability.
  - Lowers to `mbarrier.arrive.expect_tx.shared::cta.b64`, with the arrival
    token discarded: consumers wait on the phase parity, not on a token handed
    between threads. The runtime publishes no entry for it; `ops::copy_async_bulk`
    writes its own for the bulk tier.

### MBarrierWaitParity

```python
class MBarrierWaitParity(Op):
    """Effect form; block until the barrier's phase parity reaches a value.

    Attributes:
        barrier: input; smem barrier object.
        phase: input; the parity waited for.
    """

    barrier: Tensor
    phase: Tensor
```
- constraints:
  - `barrier` is smem.
  - The parity alternates `0, 1, 0, ...` across successive completions, which is
    what lets a fixed ring of barriers serve a pipeline of any length: stage `t`
    of a ring of `n` waits on parity `(t // n) & 1`.
  - It requires the target's `mbarrier.try_wait` capability.
  - Lowers to a single `mbarrier.try_wait.parity.shared::cta.b64` under a **C++**
    loop, not a PTX one: a label inside inline asm is emitted once per
    instantiation and collides as soon as two of them land in one translation
    unit, and `try_wait` already parks the warp in hardware for a bounded
    interval, so the loop is not a busy spin on the issue pipe. The runtime
    publishes no entry for it.
  - `phase` is a value and not an attribute — the parity is a function of the
    stage index — and lowers through the same scalar-expression renderer as
    `If.cond`, so a loop induction variable or a constant reaches the
    instruction and anything else is refused at codegen.

### MBarrierInvalidate

```python
class MBarrierInvalidate(Op):
    """Effect form; release the barrier's shared-memory word.

    Attributes:
        barrier: input; smem barrier object.
    """

    barrier: Tensor
```
- constraints:
  - `barrier` is smem.
  - Lowers to `mbarrier.inval.shared::cta.b64`, written at the call site: the
    runtime publishes no entry for it.
