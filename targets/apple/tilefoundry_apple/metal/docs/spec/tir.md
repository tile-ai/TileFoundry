# Metal special instruction contracts

## 1. Scope and types

The catalogue follows native APIs used by Apple MLX. Ordinary scalar math,
indices, tensor views and composite algorithms reuse core IR. Each target Op
owns ParamDef operand patterns, effects, capability and execution mesh, as in
CUDA. All tensor operands are ordinary TensorType/Layout/ShardLayout/Mesh;
there is no opaque matrix, cooperative-tensor or atomic-word IR type.

Matrix and SIMD profiles assume32 lanes. Complete quads are four-thread meshes.
Matrix and non-reduction SIMD declarations admit groups in 1..32. Reduce
admits groups in 1..256 with groups times width <=1024 and width in {4,32}.
Slots1/2/4 describe scalar or vector payloads held locally by each lane. A
runtime selector is a u16 scalar when uniform, or a lane-distributed u16 tensor
for INDEX. Its runtime value must identify a valid active lane; uniform masks
and deltas must be equal across participants.
## 2. Collective and synchronization catalogue

| Op | Contract |
| --- | --- |
| reduce | SUM/PRODUCT/MIN/MAX on f16/f32 or native8/16/32-bit integers; AND/OR/XOR on integers; ALL/ANY on bool |
| reduce, four-thread mesh | quad_sum; only SUM is admitted for this scope |
| shuffle | INDEX, BROADCAST, XOR, UP/DOWN, ROTATE_UP/DOWN; runtime selector; float XOR is valid |
| shuffle_fill | UP/DOWN with explicit source, fill, uniform delta, modulo2/4/8/16/32 |
| prefix_sum | Inclusive/exclusive addition scan |
| prefix_product | Inclusive/exclusive product scan |
| ballot | bool to low32-bit u32 mask; retained support, not found in this MLX snapshot |
| barrier | Whole declared threadgroup or aligned complete32-thread submesh; none/threadgroup/device/all memory flags |
| fence | Metal4 private system-scope seq_cst device-memory fence; no execution rendezvous |

Reduction outputs have logical shape `(groups,1[,slots])` and broadcast their
lane dimension. Scans and shuffles preserve input shape. A quad sum uses width4
in both operand layout and execution mesh, without a redundant Op scope prefix.
UP/DOWN preserve edge lanes without wrapping; rotate wraps. Fill selects from
the neighboring fill vector on crossing its modulo boundary. The delta must be
within0..modulo. Other selectors are within0..31; invalid known constants are
refused and dynamic values are rejected by the corresponding evaluator before writes.

BF16 communication preserves bits through ushort payloads in the source-derived contract.
BF16 reduction and scan arithmetic use explicit core casts to/from F32, following
MLX.64-bit/complex collectives use word/vector shuffles and core combines rather
than a fictional native reduction type. Runtime selector accesses are marked
may-access in the ISL relations; they are not fabricated fixed affine addresses.

Barrier participants come from the mesh. A device-memory flag does not turn a
threadgroup barrier into device-wide synchronization. System coherence/fence is
a distinct private Metal4 contract and does not imply an arbitrary grid barrier.

## 3. Device atomic and coherent memory

| Op | Native contract |
| --- | --- |
| atom_load | Device i32/u32/f32, relaxed load |
| atom_store | Device i32/u32/f32, relaxed store |
| atom | ADD on i32/u32/f32; AND/OR/MIN/MAX on i32/u32; returns old value |
| atom_cas | Weak relaxed compare/exchange; returns success, updates expected on failure and object on success |
| ldcoherent / stcoherent | Metal4 volatile coherent(system) device u32 access |

Backing storage is a contiguous ordinary gmem tensor; each participating thread
supplies an index and value through a one-dimensional request layout. Indices
may be i32/u32/i64 and must be nonnegative and within the backing tensor.
Atomic words must have their native alignment. Partial stores preserve the rest
of the backing tensor, so their effect is read/write. Dynamic addresses are
explicit may-access relations. There is no hidden atomic pointer type.

Float min/max and multiplication in MLX are CAS loops; narrow packed atomics
operate on u32 words and preserve neighboring fields. Complex add is two float
transactions. These are compositions, not additional indivisible instructions.
The generic MLX template mentioning ulong does not prove a native ulong fetch
operation: the retained SDK probes reject those signatures.

## 4. Matrix profiles

Public calls use the following operand order (attributes are keyword parameters).
R/W/RW denote read, write and read/write effects.

| Declaration / T.metal call | Operands | Attributes / native interface |
| --- | --- | --- |
| LdMatrix / ldmatrix | dst W, src R | dtype=f32, transpose=False; simdgroup_load |
| StMatrix / stmatrix | dst W, src R | dtype=f32, transpose=False; simdgroup_store |
| MatMul / matmul | dst W, lhs R, rhs R | dtype; simdgroup_multiply(d,a,b) |
| Mma8x8x8 / mma_m8n8k8 | dst W, lhs R, rhs R, acc R | lhs_dtype, rhs_dtype, acc_dtype, dst_dtype; simdgroup_multiply_accumulate(d,a,b,c) |
| MatMul16x32x16 / matmul_m16n32k16 | dst W, lhs R, rhs R | a_dtype, b_dtype, relaxed_precision=True; matmul2d multiply mode |
| Mma16x32x16 / mma_m16n32k16 | acc RW, lhs R, rhs R | a_dtype, b_dtype, relaxed_precision=True; matmul2d multiply_accumulate mode |
| LdMatrix16 / ldmatrix16 | dst W, src R | columns=16 or 32; cooperative_tensor.load |
| StMatrix16 / stmatrix16 | dst W, src R | columns=16 or 32; cooperative_tensor.store |

8×8 transfers use f16/f32, leading stride >=8 and batch stride >=8 times
leading stride. Each lane owns two slots, with lane l and slot s mapped to
row `4*((l//16)%2)+(l//2)%4`, column `4*((l//8)%2)+2*(l%2)+s`.
Mma8x8x8 accepts only (A,B,C,D)=(f16,f16,f16,f16) or
(f32,f32,f32,f32); defaults are f32. D and C may alias; required reads precede
the write. Mixed tuples are rejected by the normal instruction checker.

16×32×16 uses A16×16/B16×32/C16×32, SIMD32, A8/B16/C16 slots.
Allowed (A,B,C) tuples are (f16,f16,f32), (bf16,bf16,f32),
(f32,f32,f32), (f32,f16,f32), (f32,bf16,f32).
Sixteen-row transfers require row_stride >= columns and batch_stride >=
16 times row_stride, for columns in {16,32}; these are layout bindings.
Only non-transpose and explicit relaxed_precision=True are admitted: f32
multiplication may truncate mantissas. For lane l and local slot s:

```text
q = l >> 2
row = ((q & 4) | ((l >> 1) & 3)) + 8 * ((s >> 2) & 1)
col = 4 * ((q & 2) | (l & 1)) + (s & 3) + 16 * (s >> 3)
```

A interprets row/col as m/k, B as k/n and C as m/n. This is the fixed MLX
steel/gemm/nax.h profile (lines 53–59, 401–455), not an `apple.m2_pro.gpu` capability.
The TIR accumulator-first order maps to SDK run(left,right,destination).
Unsupported tuples, transpose and layouts are rejected by Pattern/check_call.

MSL leaves matrix/thread mappings unspecified and MPP layouts device-dependent.
These are source-pinned profiles, not a portable Metal ABI. Wider/tail-masked
cooperative tensors, alternate transpose modes and device-specific mappings
are not inferred from one fragment. Core views, padding and explicit tiles can
compose operations within the admitted profile; the package does not claim to
lower every MLX kernel or expose every public MPP descriptor.

## 5. Matching and evaluation

HIR inference reuses generic logical rules with necessary target storage
refinement. Candidate/check_call performs operand Pattern matching; it does not
rewrite memo layouts or require A/B to share a distribution. Each Op registers
its owned semantic handlers explicitly.

Python evaluation is logical CPU semantics. It chooses one legal atomic order
and a nonspurious CAS, models fences structurally, and returns ideal arithmetic
for NAX while keeping its relaxed-precision flag visible. These results are not
hardware rounding, coherence or arbitration references. There is no hand-written
C++ numerical oracle. Signature probes compile small MSL entry points offline;
they establish SDK acceptance, not execution or throughput on a physical device.

Sources: [MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
[MLX Metal kernels](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels).
## 6. SDK and compiler evidence

| TIR family | Actual SDK interface | Evidence level |
| --- | --- | --- |
| Mma8x8x8 | metal::simdgroup_multiply_accumulate(d,a,b,c) | SDK builtin __metal_simdgroup_matrix_8x8_multiply_accumulate; retained AIR air.simdgroup_matrix_8x8_multiply_accumulate.* |
| MatMul | metal::simdgroup_multiply(d,a,b) | SDK uses multiply-accumulate with zero C; not a separate proven machine opcode |
| Mma16x32x16 / MatMul16x32x16 | mpp::tensor_ops::matmul2d<desc, execution_simdgroup>::run(left,right,destination) | SDK __tensorops_impl_matmul2d_op_run_cooperative_f16_f16_f32 and typed variants; AIR externally_defined extern, subsequent intrinsic unknown |
| LdMatrix / StMatrix | metal::simdgroup_load/store | Public MSL API; lane mapping source-pinned |
| LdMatrix16 / StMatrix16 | cooperative_tensor.load/store | MPP library API, not proven single opcode |
| Reduce | simd_sum/product/min/max/and/or/xor/all/any; quad_sum | Public API; kind/dtype/scope limits in section 2 |
| Shuffle | simd_shuffle/broadcast/shuffle_xor/shuffle_up/shuffle_down/shuffle_rotate_up/shuffle_rotate_down | Public API, selected by kind |
| ShuffleFill | simd_shuffle_and_fill_up/down | Public API; uniform delta/modulo |
| PrefixSum / PrefixProduct | simd_prefix_inclusive_sum/product, simd_prefix_exclusive_sum/product | Public API; inclusive selects mode |
| Ballot | simd_ballot | Public API; bool input |
| Barrier | threadgroup_barrier / simdgroup_barrier | Public API; mesh selects scope |
| AtomicLoad / AtomicStore | atomic_load_explicit / atomic_store_explicit | Public API; relaxed order |
| AtomicRmw / AtomicCas | atomic_fetch_{add,and,or,min,max}_explicit / atomic_compare_exchange_weak_explicit | Public API; retained type restrictions in section 3 |
| Fence / CoherentLoad / CoherentStore | atomic_thread_fence / volatile coherent(system) device access | Private Metal4 source/SDK contract; no claimed AGX opcode |

Builtin, AIR intrinsic, externally defined library entry and machine opcode are
different evidence levels. No AGX instruction encoding is established here.
SDK headers and retained offline AIR establish signatures, not runtime speed or
availability. The 16-row library transfer contracts are static/logic evidence.


## 7. Complete construction and operand signatures

These signatures separate Op construction attributes from statement tensor
operands. `C(attrs)(operands)` below describes the two contracts; it is not
Python currying syntax. All enum names belong to `T.metal`; dtype defaults refer
to DType. Every tensor operand is a ParamDef input with the indicated effect.
An empty operand tuple means no tensor operands. The statement result is UnitType.

```text
LdMatrix(dtype=f32, transpose=False)                  (dst W, src R)
StMatrix(dtype=f32, transpose=False)                  (dst W, src R)
MatMul(dtype=f32)                                    (dst W, lhs R, rhs R)
Mma8x8x8(lhs_dtype=f32, rhs_dtype=f32,
          acc_dtype=f32, dst_dtype=f32)              (dst W, lhs R, rhs R, acc R)
MatMul16x32x16(a_dtype=f32, b_dtype=f32,
               relaxed_precision=True)              (dst W, lhs R, rhs R)
Mma16x32x16(a_dtype=f32, b_dtype=f32,
            relaxed_precision=True)                 (acc RW, lhs R, rhs R)
LdMatrix16(columns=16, dtype=f32)                    (dst W, src R)
StMatrix16(columns=16, dtype=f32)                    (dst W, src R)
Reduce(kind=ReductionKind.SUM, dtype=f32, slots=1)    (dst W, src R)
Shuffle(kind=ShuffleKind.INDEX, dtype=f32,
        slots=1, uniform=True)                      (dst W, src R, index R)
ShuffleFill(kind=ShuffleKind.UP, dtype=f32, slots=1,
            modulo=32, uniform=True)                (dst W, src R, fill R, delta R)
Ballot()                                            (dst W, src R)
PrefixSum(dtype=f32, slots=1, inclusive=False)        (dst W, src R)
PrefixProduct(dtype=f32, slots=1, inclusive=False)    (dst W, src R)
Barrier(fence=MemoryFence.THREADGROUP)               ()
Fence(fence=MemoryFence.DEVICE,
      order=AtomicOrder.SEQ_CST,
      scope=ThreadScope.SYSTEM)                     ()
AtomicLoad(dtype=u32, order=AtomicOrder.RELAXED)      (dst W, object R, index R)
AtomicStore(dtype=u32, order=AtomicOrder.RELAXED)     (object RW, value R, index R)
AtomicRmw(dtype=u32, order=AtomicOrder.RELAXED,
          kind=AtomicKind.ADD)                      (dst W, object RW, value R, index R)
AtomicCas(dtype=u32, order=AtomicOrder.RELAXED)       (success W, object RW, expected RW, desired R, index R)
CoherentLoad(dtype=u32)                              (dst W, object R, index R)
CoherentStore(dtype=u32)                             (object RW, value R, index R)
```

All attributes have declared Python types: dtype uses DType, kind/order/fence/
scope use the named enum, transpose/uniform/inclusive/relaxed_precision use bool,
and columns/slots/modulo use int. Wrong types and unsupported values fail the
existing Pattern/check_call path rather than a second Metal constructor checker.

The 8×8 transfer leading/batch strides are layout bindings, not construction
attributes. Transfer16 has no transpose attribute; dtype is f16/bf16/f32 and
columns is 16 or 32. Register fragments are RMEM; transfer memory is GMEM or
SMEM. Transfer destinations have WRITE effects, including StMatrix/StMatrix16.
The shapes and matrix mesh/layout constraints are specified in section 4.

Collective numeric types are f16/f32 and signed/unsigned 8/16/32-bit integers.
Shuffle and ShuffleFill additionally admit bf16 as transported bits. Slots are
1/2/4; ALL/ANY require slots=1 and bool. Reduce permits width 4 only for SUM,
otherwise width 32; groups are 1..256 with groups*width<=1024. Shuffle permits
nonuniform selectors only for INDEX; other kinds require uniform=True.
ShuffleFill admits only UP/DOWN, modulo 2/4/8/16/32 and uniform=True.
Ballot has fixed bool input/u32 broadcast output. Prefix operations admit numeric
types and inclusive False/True. Vector operands use the SIMD32 sharded layout;
Reduce's destination broadcasts the reduced lane axis. Invalid selector values
are checked before evaluator writes, not silently wrapped except rotate modes.

Atomic and coherent requests are RMEM rank-one tensors, one request per thread,
1..1024 requests. Object is contiguous rank-one GMEM of positive capacity;
indices have i32/u32/i64 dtype with the same request layout. Atomic dtype is
only i32/u32/f32 and order only RELAXED. AtomicKind is ADD/AND/OR/MIN/MAX;
f32 permits only ADD. CAS success is bool; expected, desired and object share
word dtype. Coherent access dtype is only u32. Dynamic out-of-range indices are
evaluator errors before writes. Store/RMW effects preserve untouched object
positions; CAS may update expected on failure. These are logical semantics,
not a promise about a hardware arbitration order.

Barrier admits NONE/THREADGROUP/DEVICE/ALL memory flags; its verifier requires
complete contiguous valid participants (whole threadgroup or aligned complete
SIMD32 group). Fence admits only DEVICE/SEQ_CST/SYSTEM and no rendezvous.
Both use the current thread execution mesh; neither invents tensor tokens.

### HIR reduction completion

HIR `tf.reduce` reduces each participant locally and leaves `Partial(tag)` on
reduced split mesh axes. Metal Reduce therefore matches the subsequent
`tf.reshard` from Partial to Broadcast, not the local Reduce. Its scalar
Partial source has logical shape `(groups, 1)`, zero local strides, and shard
attributes `(Split(0), Partial(tag))`; tag must agree with SUM/MIN/MAX.
The existing native `(groups, width[, slots])` Split source remains supported.
The Partial branch requires slots=1 and preserves width-4 SUM-only admission.
Its logical access relation is identity; the layout represents communication
between participants. The evaluator combines local ShardTensor pieces using
`to_global`; a global Partial already holds its reduced value under the public
ShardTensor contract and is copied without reducing it again.
