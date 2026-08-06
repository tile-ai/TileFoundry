"""HIR insert_slice (dynamic-update-slice): its bounds, and the in-place
loop-carry lowering exercised through a single decode step.

``insert_slice(dst, update, offsets)`` returns ``dst`` with ``update`` written
into the per-axis window at ``offsets``. A rank-1 dst takes a single scalar
start (a rank-0 integer tensor or an integer literal); a rank-N dst takes a
tuple of per-axis rank-0 offsets (literals or runtime scalars).

No corpus model calls ``insert_slice``, so this file is its whole witness. The
bounds live in two places by necessity: a literal offset is checked at typeinfer
and names the offending axis, while a runtime offset is invisible there and must
be caught by the eval guard. The decode-step tests are the GPU witness for the
in-place loop-carry lowerings (grid-region carry, ``full_like``,
``tuple_get_item``, ``cache_update``, in-place ``insert_slice`` at a runtime
scalar offset) and for the cross-CTA reshard-owned sync.
"""
from __future__ import annotations

import pytest
import torch

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call, Constant, Tuple, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.types import DType, TupleType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_F = DType.f32
_I = DType.i32
_OP = InsertSlice()

_SI64 = make_tensor_type((), DType.i64)
_SI32 = make_tensor_type((), DType.i32)


def _lit(v: int) -> Constant:
    """A compile-time literal offset (rank-0 i64 Constant)."""
    return Constant(value=v, type=_SI64)


def _rt(name: str = "p") -> Var:
    """A runtime offset (rank-0 i32 Var)."""
    return Var(type=_SI32, name=name)


def _offsets(*elems) -> Tuple:
    return Tuple(type=TupleType(fields=tuple(e.type for e in elems)), elements=tuple(elems))


def _infer_insert(dst_ty, upd_ty, offsets_expr):
    call = Call(
        type=dst_ty,
        target=InsertSlice(),
        args=(Var(type=dst_ty, name="dst"), Var(type=upd_ty, name="upd"), offsets_expr),
    )
    return TypeInferVisitor(TypeInferContext()).visit(call)


_DSL_PRELUDE = (
    "from __future__ import annotations\n"
    "from tilefoundry import func\n"
    "from tilefoundry.dsl import Tensor, tf\n"
    "\n"
)


def _eval_rankn(dst: torch.Tensor, upd: torch.Tensor, lit_offsets, runtime_axis=None):
    """Evaluate a rank-N insert_slice through the parsed DSL surface.
    ``lit_offsets`` are per-axis literals; if ``runtime_axis`` is given, that
    axis's offset is a runtime rank-0 i32 param carrying
    ``lit_offsets[runtime_axis]`` instead."""
    d = ", ".join(str(int(x)) for x in dst.shape)
    u = ", ".join(str(int(x)) for x in upd.shape)
    extra_params, inputs, elems = [], [dst, upd], []
    for ax, o in enumerate(lit_offsets):
        if ax == runtime_axis:
            extra_params.append(f'o{ax}: Tensor[(), "i32"]')
            inputs.append(torch.tensor(int(o), dtype=torch.int32))
            elems.append(f"o{ax}")
        else:
            elems.append(str(int(o)))
    extra = "".join(f", {p}" for p in extra_params)
    src = (
        _DSL_PRELUDE + "@func\n"
        f'def ins(dst: Tensor[({d}), "f32"], upd: Tensor[({u}), "f32"]{extra}) -> Tensor[({d}), "f32"]:\n'
        f"    return tf.insert_slice(dst, upd, ({', '.join(elems)}))\n"
    )
    return evaluate(parse_script(src), *inputs, device="cpu")


def _ref_scatter(dst, upd, offsets):
    import builtins  # noqa: PLC0415 -- `from tf import *` shadows the builtin `slice`

    ref = dst.clone()
    sl = tuple(builtins.slice(o, o + upd.shape[ax]) for ax, o in enumerate(offsets))
    ref[sl] = upd
    return ref

CASES = [
    # update rank must equal dst rank.
    TypeInferCase(
        "rank_mismatch_rejected",
        _OP,
        (make_tensor_type((8,), _F), make_tensor_type((2, 4), _F), make_tensor_type((), _I)),
        ExpectedError("update rank .* must equal dst rank"),
    ),
    # A bare scalar offset applies only to a rank-1 dst; a multi-D dst needs a
    # per-axis offset tuple.
    TypeInferCase(
        "nd_scalar_offset_rejected",
        _OP,
        (make_tensor_type((4, 8), _F), make_tensor_type((1, 8), _F), make_tensor_type((), _I)),
        ExpectedError("per-axis offset tuple"),
    ),
    # offsets must be an integer scalar.
    TypeInferCase(
        "offsets_dtype_rejected",
        _OP,
        (make_tensor_type((8,), _F), make_tensor_type((3,), _F), make_tensor_type((), _F)),
        ExpectedError("offsets must be an integer scalar"),
    ),
    # A Partial dst may only be written by an equally-Partial update; the
    # update side is symmetric and reported by name the same way.
    TypeInferCase(
        "partial_dst_plain_update_rejected",
        _OP,
        (
            make_shard_tensor_type((8,), mesh=make_mesh((4,)), attrs=(Partial("sum"),)),
            make_tensor_type((3,), _F),
            make_tensor_type((), _I),
        ),
        ExpectedError("dst carries a Partial"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_insert_slice_typeinfer(case):
    run_typeinfer_case(case)


# ── rank-N per-axis offset tuple (typeinfer) ──────────────────────────────

def test_insert_slice_rankn_static_oob_names_axis():
    """An all-literal offset that puts the window past dst on one axis is
    rejected at typeinfer, and the error names the offending axis. A negative
    literal is the same check from the other side, so it is not asserted
    separately."""
    with pytest.raises(VerifyError, match="axis 1"):
        _infer_insert(
            make_tensor_type((1, 16512, 512), _F),
            make_tensor_type((1, 1, 512), _F),
            _offsets(_lit(0), _lit(16512), _lit(0)),  # 16512 + 1 > 16512 on axis 1
        )


def test_insert_slice_tuple_len_must_equal_rank():
    with pytest.raises(VerifyError, match="tuple length"):
        _infer_insert(
            make_tensor_type((1, 16512, 512), _F), make_tensor_type((1, 1, 512), _F), _offsets(_lit(0), _lit(0))
        )


# ── rank-N per-axis offset tuple (evaluation) ─────────────────────────────

def test_insert_slice_rankn_eval_matches_reference_scatter():
    """A rank-3 window at ``(0, P%128, 0)`` (the middle offset a runtime member)
    evaluates to the same tensor as a reference scatter. This is also where
    typeinfer accepting a runtime offset member is shown to be sound: the
    literal members are still bounds-checked, the runtime one is deferred to
    here."""
    torch.manual_seed(0)
    dst = torch.randn(1, 16512, 512)
    upd = torch.randn(1, 1, 512)
    p = 640 % 128  # a runtime middle-axis offset
    out = _eval_rankn(dst, upd, (0, p, 0), runtime_axis=1)
    torch.testing.assert_close(out, _ref_scatter(dst, upd, (0, p, 0)))


def test_insert_slice_rankn_eval_runtime_oob_raises():
    """A runtime offset member that puts the window out of bounds is caught by
    the eval guard (typeinfer cannot see the runtime value)."""
    dst = torch.zeros(1, 8, 4)
    upd = torch.zeros(1, 3, 4)
    with pytest.raises(ValueError, match="out of bounds"):
        _eval_rankn(dst, upd, (0, 6, 0), runtime_axis=1)  # 6 + 3 > 8 on axis 1


# ── single decode step: in-place carry lowering + reshard-owned sync ──────

from tilefoundry import func, module  # noqa: E402
from tilefoundry.dsl import Mesh, Tensor, Topology  # noqa: E402
from tilefoundry.dsl.storage import gmem  # noqa: E402
from tilefoundry.dsl.tf import *  # noqa: E402,F401,F403
from tilefoundry.ir.tir.stmts import (  # noqa: E402
    Evaluate,
    For,
    LetStmt,
    MeshScope,
    Sequential,
)
from tilefoundry.ir.types.shard import Layout as ShardMeshLayout  # noqa: E402
from tilefoundry.passes.transforms import HirToTirPass  # noqa: E402

_DEC_D = 4
_DEC_STEPS = 3
_CACHE_CAP = 4
_KV_HEADS = 1
_HEAD_DIM = 4


@module(entry="decode_step")
class _DecodeStep:
    """A single decode step exercising the in-place loop-carry lowerings: a
    two-carry grid region (output accumulator + running total → a tuple, so
    ``tuple_get_item``), ``full_like`` inits, an in-place ``insert_slice`` write
    at a dynamic scalar offset, and a rank-4 ``cache_update`` KV write."""
    topologies = (Topology("thread", 1),)


    @func
    def decode_step(
        x: Tensor[(_DEC_D,), "f32"],
        v: Tensor[(1,), "f32"],
        kcache: Tensor[(1, _CACHE_CAP, _KV_HEADS, _HEAD_DIM), "f32"],
        kin: Tensor[(1, 1, _KV_HEADS, _HEAD_DIM), "f32"],
        cur: Tensor[(1,), "i32"],
        spos: Tensor[(1,), "i32"],
        off: Tensor[(), "i32"],
    ):
        with Mesh(Topology("thread", 1), (1,), ("t",)) as m:
            xr = reshard(x, (_DEC_D @ m.t,), "rmem")
            vr = reshard(v, (1 @ m.t,), "rmem")
            acc = full_like(xr, 0.0)
            cnt = full_like(xr, 0.0)
            for i in tile(_DEC_STEPS):
                acc = insert_slice(acc, vr, off)
                cnt = add(cnt, xr)
            result = add(acc, cnt)
            kc = cache_update(kcache, cur, spos, kin)
            return (reshard(result, (_DEC_D @ m.t,), "gmem"), kc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_step_matches_torch() -> None:
    """The decode step compiles and runs on GPU; the accumulator write at a
    dynamic offset and the KV cache update match a torch reference."""
    import tilefoundry  # noqa: PLC0415

    rm = tilefoundry.compile(_DecodeStep, target=CudaTarget("nvidia.h200_sxm"))
    x = torch.randn(_DEC_D, device="cuda")
    v = torch.randn(1, device="cuda")
    kcache = torch.zeros(1, _CACHE_CAP, _KV_HEADS, _HEAD_DIM, device="cuda")
    kin = torch.randn(1, 1, _KV_HEADS, _HEAD_DIM, device="cuda")
    cur = torch.tensor([1], dtype=torch.int32, device="cuda")
    spos = torch.tensor([1], dtype=torch.int32, device="cuda")
    out = torch.empty(_DEC_D, device="cuda")
    kc_out = torch.empty_like(kcache)
    off = 2
    rm(x, v, kcache, kin, cur, spos, off, out, kc_out)
    torch.cuda.synchronize()

    exp = _DEC_STEPS * x.clone()
    exp[off] = exp[off] + v[0]
    assert torch.allclose(out, exp, rtol=1e-4, atol=1e-4), (out - exp).abs().max()
    exp_kc = kcache.clone()
    exp_kc[:, 1:2] = kin
    assert torch.allclose(kc_out, exp_kc, rtol=1e-4, atol=1e-4), (kc_out - exp_kc).abs().max()


def _lower(mod):
    return HirToTirPass().run(mod).functions[0]


def _walk(node, in_loop, out):
    if isinstance(node, Sequential):
        for s in node.body:
            _walk(s, in_loop, out)
    elif isinstance(node, MeshScope):
        _walk(node.body, in_loop, out)
    elif isinstance(node, For):
        _walk(node.body, True, out)
    elif isinstance(node, LetStmt):
        out.append((in_loop, "let", node.var, node.value))
        _walk(node.body, in_loop, out)
    elif isinstance(node, Evaluate):
        out.append((in_loop, "eval", None, node))


def _op_of(value):
    return getattr(value, "target", None) or getattr(value, "callable", None)


# A cross-CTA ownership-change reshard (split axis 0 to split axis 1) at the
# output position (the returned value). The reshard-owned grid fence must not
# be bypassed by the output-sink lowering.
@func(topologies=(Topology("cta", 2),))
def _cross_cta_reshard_output(a: Tensor[(2, _DEC_D), "f32"]) -> Tensor[(2, _DEC_D), "f32"]:
    with Mesh(topology="cta", layout=ShardMeshLayout(shape=(2,), strides=(1,))) as cta:
        g1 = reshard(a, layout=(2 @ cta, _DEC_D), storage=gmem)
        return reshard(g1, layout=(2, _DEC_D @ cta), storage=gmem)


def test_cross_cta_reshard_owned_sync() -> None:
    """An output-position cross-CTA reshard (ownership change) still lowers to
    sync-then-reshard: the grid sync is emitted before the output copy, proving
    the output-sink path routes through the same reshard-owned fence as an
    intermediate reshard."""
    pf = _lower(_cross_cta_reshard_output)
    nodes = []
    _walk(pf.body, False, nodes)

    kinds = [
        type(_op_of(val)).__name__
        for in_loop, kind, var, val in nodes
        if _op_of(val) is not None
    ]
    assert "Sync" in kinds, f"no reshard-owned grid sync emitted: {kinds}"
    # The sync fences before the output reshard's copy.
    assert kinds.index("Sync") < len(kinds) - 1 - kinds[::-1].index("Copy")


# ── rank-N insert_slice end-to-end on GPU: per-axis window + coords ────────
#
# A non-contiguous per-axis window (partial inner axis) written at a dynamic,
# non-zero middle-axis coordinate. This is the numerical gate for the rank-N
# codegen: it fails if the emitter drops trailing coordinates (writes at the
# wrong axis) or flat-collapses the window shape (a partial inner axis is not
# contiguous in the flattened buffer). The offset is the loop induction
# variable, so a per-iteration dynamic offset is covered here too.
_NW_A, _NW_B, _NW_C = 1, 4, 6
_NW_UB, _NW_UC = 2, 3  # window extent on axis 1 / axis 2 (partial: 3 of 6)
_NW_STEPS = 2


@module(entry="nd_window")
class _NdWindow:
    """A loop-carried rank-3 in-place ``insert_slice`` writing a non-trivial,
    non-contiguous window (full axis 0, window 2 on axis 1, partial 3-of-6 on
    axis 2) at the induction variable as the middle-axis tile coordinate."""
    topologies = (Topology("thread", 1),)


    @func
    def nd_window(
        base: Tensor[(_NW_A, _NW_B, _NW_C), "f32"],
        v: Tensor[(_NW_A, _NW_UB, _NW_UC), "f32"],
    ):
        with Mesh(Topology("thread", 1), (1,), ("t",)) as m:
            br = reshard(base, (_NW_A, _NW_B, _NW_C @ m.t), "rmem")
            vr = reshard(v, (_NW_A, _NW_UB, _NW_UC @ m.t), "rmem")
            acc = full_like(br, 0.0)
            for i in tile(_NW_STEPS):
                acc = insert_slice(acc, vr, (0, i, 0))
            return reshard(acc, (_NW_A, _NW_B, _NW_C @ m.t), "gmem")




@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_insert_slice_rankn_gpu_oracle() -> None:
    """The rank-N in-place ``insert_slice`` runs on GPU and matches a torch
    scatter reference: a non-contiguous window at a dynamic, non-zero
    middle-axis coordinate."""
    import tilefoundry  # noqa: PLC0415

    rm = tilefoundry.compile(_NdWindow, target=CudaTarget("nvidia.h200_sxm"))
    base = torch.randn(_NW_A, _NW_B, _NW_C, device="cuda")
    v = torch.randn(_NW_A, _NW_UB, _NW_UC, device="cuda")
    out = torch.empty(_NW_A, _NW_B, _NW_C, device="cuda")
    rm(base, v, out)
    torch.cuda.synchronize()

    exp = torch.zeros(_NW_A, _NW_B, _NW_C, device="cuda")
    for i in range(_NW_STEPS):
        exp[:, _NW_UB * i : _NW_UB * i + _NW_UB, 0:_NW_UC] = v
    assert torch.allclose(out, exp, rtol=1e-4, atol=1e-4), (out - exp).abs().max()
