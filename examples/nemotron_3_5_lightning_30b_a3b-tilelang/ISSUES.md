# What got in the way

Things hit while taking NVIDIA-Nemotron-3.5-Lightning-30B-A3B to a one-launch
decode step. Two parts: **TileFoundry itself**, and the **TileLang** backend
underneath it. Every entry has a minimal repro; the ones that can be run carry
their command line and their actual output.

Repro files are under `repro/` (TileFoundry) and `kbench/` (TileLang).

| # | in one line | repro | what it blocked |
|---|---|---|---|
| **TF-1** | a slice start carrying a mesh index will not evaluate | `repro/mesh_slice_start.py` | **long-context attention cannot be `check`ed** |
| TF-2 | callee dispatch now passes `--dim`; entry dispatch still needs a tuple return annotation | `repro/specialize_through_call.py` | the entry cannot carry the dispatch; the callee route is fixed |
| TF-3 | TF-1's error carried no file, line or op | same | locating it |
| TF-4 | checking one leaf materialises the whole Module's weights | `repro/leaf_weights.py` | no leaf of this model can be checked |
| TF-5 | `--inputs random` builds states the model cannot be in, and reports an out-of-range first | — | had to dump real activations |
| TF-6 | `--input` and `--inputs real` cannot both be given | — | real weights *and* real activations |
| TF-7 | `CudaTarget(arch=, num_sms=)` is not the actual signature | — | documentation |
| **TL-1** | descriptor TMA cannot use a device-bound pointer; a constant coefficient falls out of the bulk path | `kbench/tma2d.py` | the rule is undocumented; nine address shapes were tried |
| **TL-2** | `T.gemm`'s layout inference reaches through `T.view` to the buffer underneath | `kbench/tma2dbuf.py` | one allocation cannot be both a bulk-copy target and an operand |
| TL-3 | a gemm benchmark whose operands do not change gets hoisted out of the loop and measures the wrong thing | `kbench/lingemm.py` vs `blkloop.py` | 6x |
| **TL-4** | the shared allocator does not reuse disjoint lifetimes, and fails only at launch | `kbench/reuse2.py` | the whole budget had to be rearranged |
| TL-5 | at M=16: wgmma refuses, `T.reduce_max` will not lower, a 3-D fragment cannot be an output, fragment→fragment copy conflicts on layout | `kbench/wgattn.py` `fraga.py` | the natural shape of GQA decode |
| TL-6 | `ThreadSync` hoists a barrier out of an `if` and says itself that this is not a fix | — | had to become a select on pointers |
| TL-7 | `T.copy`'s default instruction is 23% slower than `cp_async` | — | a one-line change |
| TL-8 | the eager builder rejects a Python-level loop over a tuple | — | manual unrolling |

---

## I. TileFoundry

### TF-1 (the one that matters) a slice start carrying a mesh index will not evaluate

**Repro:** `repro/mesh_slice_start.py`

```bash
tilefoundry check repro/mesh_slice_start.py:Fixed   --inputs random --out output --fn nan_inf
tilefoundry check repro/mesh_slice_start.py:Strided --inputs random --out output --fn nan_inf
```

The two Modules differ by one line:

| | `b0` | result |
|---|---|---|
| `Fixed` | `b0 = t` | PASS |
| `Strided` | `b0 = t + m.w * BLK` | `shape '[]' is invalid for input of size 4` |

The `4` is the width of that mesh axis.

**Why it matters.** "Split a long reduction over a worker axis, and let unit *w*
walk every *W*-th block" is the most basic way to write this kind of program —
and for long-context attention it is the only placement with enough parallel
units. In HIR that is `for t in tile(CF, BLK * W)` with `b0 = t + kv.w * BLK`,
which is exactly the shape the evaluator cannot run.

The consequence is that this model's `check` **only runs at `ctx_full = 0`**,
where the tile loop runs zero times and the path is never taken. The
long-context arm's numerical correctness cannot be established by `check`; it
rests on mega-against-op-by-op and on the token-for-token comparison with
`transformers`.

**As of `74abc97` the message has become explicit** — it now names the file,
line, column, binding and op, and points at the spec:

```
evaluator: repro/mesh_slice_start.py:62:26: binding=<unnamed> op=Local:
Local on a Split axis is not modelled: evaluation runs one mesh participant
(docs/spec/evaluator.md section 6)
```

That is TF-3 fixed (below). TF-1 itself is a stated boundary rather than a
crash, and is being worked on upstream.

### TF-2 there is nowhere to put the dispatch that the real entry can reach

**Repro:** `repro/specialize_through_call.py`

```bash
tilefoundry check repro/specialize_through_call.py:Direct   --inputs random --dim n=64 --out output --fn nan_inf
tilefoundry check repro/specialize_through_call.py:ToCallee --inputs random --dim n=64 --out output --fn nan_inf
python repro/specialize_through_call.py
```

| where it is put | result |
|---|---|
| `Direct`: the entry calls one variant's body directly | PASS |
| `ToCallee`: variants on the callee, entry calls the prototype | PASS |
| `ToEntry`: variants on the entry, but the entry returns several tensors | `HIR pass prototype requires a return annotation` |

**a. It now passes through — fixed.** Specialization selects a callee's variant
from the caller's `--dim` bindings and rebuilds through that implementation.
`check` and `analyze` can therefore both reach a dispatch prototype from the
entry.

**Fixed in `#145`** — `ToCallee` now passes both commands shown in its repro.

**b. Moving it to the entry does not work either.** The shape
`tilefoundry tutorial authoring` demonstrates is variants hung on the entry — but
a prototype must carry a return annotation, and the grammar for a return type is
`tensor | scalar-type`, with no tuple. A decode step returns logits *and* every
layer's new state (59 outputs in this model), so that annotation cannot be
written.

**The way around it, as shipped** (`model.py`): `attend` and its two
`DimVarRangePat` variants are written out as the statement of the intent, but
`decode_step` calls `attend_by_context` rather than `attend`. The dispatch can
therefore be checked on its own (`attention.py:AttnRuntime.attend`), while the
entry does not go through it.

### TF-3 the error did not say where — **fixed**

TF-1's output used to be a single line, `shape '[]' is invalid for input of size
4`: no file, no line number, no op name, and no word on which slice came out
empty. In a 3900-line HIR that locates nothing. TF-2a's message was much better
(it named `'pick'`) and stood as the counter-example.

**Fixed in `#141`** — see the message quoted under TF-1.

### TF-4 checking one leaf materialises the whole Module's weights

```bash
tilefoundry check runtime_model.py:Nemotron35Lightning30BA3BRuntime.attend \
    --inputs random --dim ctx_full=0 --dim ctx_tail=128 --out output --fn nan_inf
```

```
CUDA out of memory. Tried to allocate 27.36 GiB. GPU 0 has a total capacity of
139.80 GiB of which 605.12 MiB is free.
```

`attend` declares no ConstTensor at all — its parameters are only
qg / k_cache / v_cache / k_tail / v_tail — but `check` draws all 474 weights the
Module declares, once on the semantic side and once on the runtime side: about
60 GB twice for this model, which an H200 cannot hold. **So no leaf function of
this model can be checked.**

The way around it is to make that leaf a Module of its own with no weights
(`attention.py`) and give it a twin — but then what is checked is not the
function inside the main twin.

`repro/leaf_weights.py` is the same shape in miniature (turn `BIG_GB` up until
it bursts).

### TF-5 `--inputs random` builds states the model cannot be in

Running the whole decode step with `--inputs random` first gives

```
cur_pos + s (5) exceeds cache capacity 1
```

and after that, whole-model mismatch. The cause is that each activation is drawn
**independently**: `cur_pos` has to index the K/V tail, the conv and SSM states
have to be what the previous layer wrote, and a hidden row out of distribution
makes the router's top-6 ordering nothing like the checkpoint's. The tool's own
warning does say this, but **what surfaces first is an unreadable
out-of-range**, not "these activations constrain each other".

The way around it (`dump_acts.py`): decode N steps for real, save each of the 73
activations, and feed them with `--input`. Suggestion: either let
`--inputs random` respect the relationships already declared between parameters
(`cur_pos` against the cache dimension is one), or refuse on a model like this
and point at `--input`.

### TF-6 `--input` and `--inputs real` cannot both be given

`--input` (a few real activations) together with `--inputs real --ckpt` (real
weights) is refused: `give exactly one form`. For this model the two are
orthogonal — weights come from the checkpoint, activations from a real run — and
right now it has to be one or the other.

### TF-7 the constructor signature in the docs is not the real one

`CudaTarget(arch="sm_90a", num_sms=132)` → `CudaTarget.__init__() got an
unexpected keyword argument 'num_sms'`. What is actually wanted is
`CudaTarget("nvidia.h200_sxm")`.

---

## II. TileLang (the backend)

In the order they were hit. All timings on an H200 with the SM clock at 1500 MHz.

### TL-1 descriptor TMA cannot use a pointer only known on the device

```
Check failed: (result.supported) is false: Descriptor-based TMA cannot use global
base pointer `src` because it is bound inside the device function body.
TensorMap descriptors are encoded on the host; use plain T.copy to allow a
descriptorless or synchronous fallback.
```

A one-launch mega kernel can only put the K/V cache addresses in an int64 table
(474 parameters do not fit in a signature), so every cache access goes through
`T.make_tensor_from_addr`. Pointers like that cannot enter a descriptor.

**Repro:** `kbench/tma2d.py` lays out nine address shapes. The conclusion is that
the 1-D bulk path requires the block index to be multiplied by a **runtime**
value:

| address | result |
|---|---|
| no block index | FAIL |
| `bx // 2`, `bx % 2 * N` | FAIL |
| `by * N` (constant coefficient) | FAIL |
| `bx * st`, `by * st`, `bx*st + by*N + _b*N` (runtime coefficient) | OK |

The error itself is good — it says why and offers a way out — but **the rule that
a constant coefficient falls out of the bulk path is written down nowhere**; it
took nine probes to find.

### TL-2 `T.gemm`'s layout inference reaches through `T.view` to the buffer underneath

**Repro:** `kbench/tma2dbuf.py`

To make one shared allocation both a bulk-copy target (wants 1-D) and a gemm
operand (wants 2-D), both directions were tried:

| declared | used as | result |
|---|---|---|
| 1-D buffer, 2-D view | the view as a gemm operand | `The dimension of Buffer "kvf" with shape (32768,) should be at least 2` |
| 2-D buffer (default layout), 1-D view | the view as a tma_copy target | TL-1's descriptor error |
| 2-D buffer + **a linear layout pinned by hand**, 1-D view | both | **OK** (rel_l2 1.23e-7) |

The first row is the bug: `make_swizzled_layout` receives the `kvf` underneath
the view (`_get_buffer_info` returns `.buffer` for a BufferRegion) and so demands
`shape.size() >= 2` of a 1-D buffer. The second row shows the other half — gemm's
layout inference **propagates back** onto the buffer's declaration, turning a
target that could have gone bulk into a descriptor target — and the error is
raised at the TMA line, a long way from the gemm that actually caused it.

### TL-3 a linear layout makes gemm 6x slower, and the first measurement of it is wrong

**Repro:** `kbench/lingemm.py` (operands do not change) against
`kbench/blkloop.py` (operands stream in from memory)

| | ns / block | TFLOP/s |
|---|---|---|
| `lingemm` (operands constant across the REPS loop), linear | 47.9 | 43.8 |
| `lingemm`, swizzled | 11.9 | 176.9 |
| `blkloop` (operands change every block), linear | 1630 | — |
| `blkloop`, swizzled | 849 | — |

The first two rows are a **false reading**: with the operands unchanging, the
compiler hoists the whole shared→register load out of the loop and what is timed
is the mma instruction alone. The real gap is the third and fourth rows, 934 ns
against 154 ns per block. This is not a bug so much as a warning about how to
write the harness — but the cost really is in the **load**, not in the multiply.

### TL-4 the shared allocator does not reuse disjoint lifetimes, and fails at launch

**Repro:** `kbench/reuse2.py`

```
REUSE FAIL 304 KB declared: Failed to set the allowed dynamic shared memory size to 311296
```

A 176 KB arena and 128 KB of attention operands, used alternately in a loop and
never live at the same time. The allocator simply adds them. And: **it compiles**
— `build()` returns normally and only `kernel(...)` fails, so the constraint is
not known until the very end.

Related: the dynamic shared available per block is
`shared_memory_per_block_optin` (232448) **minus 1024 reserved by the driver**.
Asking for 231472, which is less than 232448, still fails — by those 48 bytes.
That reservation is not documented anywhere.

### TL-5 several things at M=16

The natural shape of GQA decode is M=16: sixteen query heads in a KV group, one
row per token.

* `T.wgmma_gemm` refuses M=16 (WGMMA's M is fixed at 64). Padding to 64 is 25%
  utilisation and not worth it; `T.gemm` picks `mma.m16n8k16` and measures
  176.9 TFLOP/s, 3.4x the CUDA-core peak.
* `T.reduce_max` **will not lower** on a (16,128) fragment: the eight warps
  divide the N axis, and a row reduction has no form to project onto.
  `GemmWarpPolicy.FullRow` does not help. Softmax had to go around through
  shared.
* A 3-D fragment as a gemm output → `TensorCoreIntrinEmitter.make_mma_store_layout`
  TypeError. Two 2-D fragments are fine.
* `T.copy` from fragment to fragment (accumulator layout → A-operand layout) →
  `Layout infer conflict between sf and pf0`. It has to go through shared.

### TL-6 `ThreadSync` hoists `__syncthreads()` out of an `if`, then says that is not a fix

```
Warning: [ThreadSync] Hoisting sync out of an if whose condition is not safe for an
in-if sync. This is not a fix: both ends of the conflict are inside the branch, so
the hoisted barrier no longer separates them and the race remains.
```

The warning is excellent — it says outright that it did not fix it, and points at
`T.assume`. The problem is that it **went ahead anyway**, and the generated
kernel carries a known race. The way around it is not to put a copy inside a
branch: in this kernel, choosing between cache and tail became a
`T.if_then_else` on the **pointer** — one select, no branch.

### TL-7 the instruction `T.copy` picks by default is much worse than `cp_async`

Same kernel, only `prefer_instruction` changed:

| | ctx 65536 | ctx 262080 |
|---|---|---|
| `prefer_instruction="cp_async"` | 3.747 ms | **4.319 ms** |
| default (TileLang chooses) | 3.991 ms | 5.314 ms |
| `cp_async` + `evict_first` | 3.766 ms | 4.325 ms |

23% at 262080. The default choice is wrong for global→shared when the source is a
`make_tensor_from_addr`.

### TL-8 the eager builder rejects a Python-level loop over a tuple

```python
for w, row in enumerate((KROW, VROW)):   # Invalid for loop, got <enumerate object>
for w, row in ((0, KROW), (1, VROW)):    # Invalid for loop, got ((0, 0), (1, 128))
```

`expect one of the following: range, T.serial, T.grid, T.parallel, T.vectorized,
T.unroll, T.thread_binding`. A Python-level loop over a constant tuple has to be
unrolled by hand. `range` works, so this looks like something that could be
relaxed.

---

## III. One piece of good news

`T.copy` (descriptorless, through cp.async) is **as fast as bulk TMA** on this
shape:

| | TB/s |
|---|---|
| `T.tma_copy` into a linear buffer | 4.140 |
| `T.copy(prefer_instruction="cp_async")` into a swizzled buffer | 4.152 |

So the whole chain of TL-1 and TL-2 — table pointer, therefore bulk only,
therefore a linear layout only, therefore gemm 6x slower — can be stepped around
entirely: use `T.copy` straight into the layout gemm chooses for itself. **At no
cost.**
