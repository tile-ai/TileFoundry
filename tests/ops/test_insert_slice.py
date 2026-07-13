"""HIR insert_slice (dynamic-update-slice) typeinfer + eval, plus the in-place
loop-carry lowering exercised through a single decode step.

``insert_slice(dst, update, offsets)`` returns ``dst`` with ``update`` written
into the per-axis window at ``offsets``. A rank-1 dst takes a single scalar
start (a rank-0 integer tensor or an integer literal); a rank-N dst takes a
tuple of per-axis rank-0 offsets (literals or runtime scalars). The decode-step
tests exercise the loop-carry lowering (grid-region carry, full_like,
tuple_get_item, cache_update, in-place insert_slice) and the cross-CTA
reshard-owned sync.
"""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
    ten,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.tuple import Tuple
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_F = DType.f32
_I = DType.i32
_OP = InsertSlice()

_SI64 = TensorType(shape=(), dtype=DType.i64, layout=None, storage="gmem")
_SI32 = TensorType(shape=(), dtype=DType.i32, layout=None, storage="gmem")


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


def _eval_rankn(dst: torch.Tensor, upd: torch.Tensor, lit_offsets, runtime_axis=None):
    """Evaluate a rank-N insert_slice with an offset tuple. ``lit_offsets`` are
    per-axis literals; if ``runtime_axis`` is given, that axis's offset is a
    runtime rank-0 param carrying ``lit_offsets[runtime_axis]`` instead."""
    from dataclasses import replace  # noqa: PLC0415

    dst_p = Var(type=TensorType(shape=tuple(dst.shape), dtype=_F, layout=None, storage="gmem"), name="dst")
    upd_p = Var(type=TensorType(shape=tuple(upd.shape), dtype=_F, layout=None, storage="gmem"), name="upd")
    params = [dst_p, upd_p]
    inputs = [dst, upd]
    elems = []
    for ax, o in enumerate(lit_offsets):
        if ax == runtime_axis:
            v = Var(type=_SI32, name=f"o{ax}")
            params.append(v)
            inputs.append(torch.tensor(o, dtype=torch.int32))
            elems.append(v)
        else:
            elems.append(_lit(o))
    offsets = _offsets(*elems)
    call = Call(type=dst_p.type, target=InsertSlice(), args=(dst_p, upd_p, offsets))
    rt = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=rt)
    fn = Function.build(name="ins", params=tuple(params), body=call, return_type=rt)
    return evaluate(fn, *inputs, device="cpu")


def _ref_scatter(dst, upd, offsets):
    import builtins  # noqa: PLC0415 -- `from tf import *` shadows the builtin `slice`

    ref = dst.clone()
    sl = tuple(builtins.slice(o, o + upd.shape[ax]) for ax, o in enumerate(offsets))
    ref[sl] = upd
    return ref

CASES = [
    # 1-D window with a rank-0 scalar offset: returns dst's type unchanged.
    TypeInferCase(
        "returns_dst_type",
        _OP,
        (ten((8,), _F), ten((3,), _F), ten((), _I)),
        ten((8,), _F),
    ),
    # A full-width update (same extent as dst) is in bounds.
    TypeInferCase(
        "full_width_update_ok",
        _OP,
        (ten((8,), _F), ten((8,), _F), ten((), _I)),
        ten((8,), _F),
    ),
    # A rank-0 scalar offset is the canonical 1-D surface.
    TypeInferCase(
        "offsets_scalar_ok",
        _OP,
        (ten((8,), _F), ten((3,), _F), ten((), _I)),
        ten((8,), _F),
    ),
    # An integer literal is carried as an i64 scalar and accepted.
    TypeInferCase(
        "offsets_i64_literal_ok",
        _OP,
        (ten((8,), _F), ten((3,), _F), ten((), DType.i64)),
        ten((8,), _F),
    ),
    # update rank must equal dst rank.
    TypeInferCase(
        "rank_mismatch_rejected",
        _OP,
        (ten((8,), _F), ten((2, 4), _F), ten((), _I)),
        ExpectedError("update rank .* must equal dst rank", exc=TypeError),
    ),
    # A bare scalar offset applies only to a rank-1 dst; a multi-D dst needs a
    # per-axis offset tuple.
    TypeInferCase(
        "nd_scalar_offset_rejected",
        _OP,
        (ten((4, 8), _F), ten((1, 8), _F), ten((), _I)),
        ExpectedError("per-axis offset tuple", exc=TypeError),
    ),
    # A rank-1 vector offset is not a rank-0 scalar start for the 1-D case.
    TypeInferCase(
        "offsets_vector_rejected",
        _OP,
        (ten((8,), _F), ten((3,), _F), ten((2,), _I)),
        ExpectedError("offsets must be a rank-0 scalar start", exc=TypeError),
    ),
    # A one-element ``(1,)`` offset is the legacy spelling and is rejected —
    # the canonical 1-D offset is a rank-0 scalar.
    TypeInferCase(
        "offsets_one_vector_rejected",
        _OP,
        (ten((8,), _F), ten((3,), _F), ten((1,), _I)),
        ExpectedError("offsets must be a rank-0 scalar start", exc=TypeError),
    ),
    # A total-size-one multi-dim ``(1, 1)`` offset is also rejected.
    TypeInferCase(
        "offsets_multi_one_rejected",
        _OP,
        (ten((8,), _F), ten((3,), _F), ten((1, 1), _I)),
        ExpectedError("offsets must be a rank-0 scalar start", exc=TypeError),
    ),
    # offsets must be an integer scalar.
    TypeInferCase(
        "offsets_dtype_rejected",
        _OP,
        (ten((8,), _F), ten((3,), _F), ten((), _F)),
        ExpectedError("offsets must be an integer scalar", exc=TypeError),
    ),
    # dst / update dtype must match.
    TypeInferCase(
        "dtype_mismatch_rejected",
        _OP,
        (ten((8,), _F), ten((3,), DType.bf16), ten((), _I)),
        ExpectedError("dst/update dtype mismatch", exc=TypeError),
    ),
    # A statically over-long update is rejected.
    TypeInferCase(
        "static_overlong_update_rejected",
        _OP,
        (ten((8,), _F), ten((10,), _F), ten((), _I)),
        ExpectedError("exceeds dst extent", exc=TypeError),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_insert_slice_typeinfer(case):
    run_typeinfer_case(case)


# ── rank-N per-axis offset tuple (typeinfer) ──────────────────────────────

def test_insert_slice_rankn_tuple_returns_dst_type():
    """A rank-3 window with an all-literal in-bounds offset tuple returns the
    dst type unchanged."""
    out = _infer_insert(
        ten((1, 16512, 512), _F), ten((1, 1, 512), _F), _offsets(_lit(0), _lit(5), _lit(0))
    )
    assert out.shape == (1, 16512, 512) and out.dtype == _F


def test_insert_slice_rankn_static_oob_names_axis():
    """An all-literal offset that puts the window past dst on one axis is
    rejected at typeinfer, and the error names the offending axis."""
    with pytest.raises(TypeError, match="axis 1"):
        _infer_insert(
            ten((1, 16512, 512), _F),
            ten((1, 1, 512), _F),
            _offsets(_lit(0), _lit(16512), _lit(0)),  # 16512 + 1 > 16512 on axis 1
        )


def test_insert_slice_rankn_negative_literal_rejected():
    with pytest.raises(TypeError, match="axis 1"):
        _infer_insert(
            ten((1, 16512, 512), _F),
            ten((1, 1, 512), _F),
            _offsets(_lit(0), _lit(-1), _lit(0)),
        )


def test_insert_slice_rankn_runtime_member_deferred():
    """A runtime offset member is not statically checkable; typeinfer accepts it
    (deferred to eval) while the literal members are still bounds-checked."""
    out = _infer_insert(
        ten((1, 16512, 512), _F), ten((1, 1, 512), _F), _offsets(_lit(0), _rt(), _lit(0))
    )
    assert out.shape == (1, 16512, 512)


def test_insert_slice_tuple_len_must_equal_rank():
    with pytest.raises(TypeError, match="tuple length"):
        _infer_insert(
            ten((1, 16512, 512), _F), ten((1, 1, 512), _F), _offsets(_lit(0), _lit(0))
        )


# ── rank-N per-axis offset tuple (evaluation) ─────────────────────────────

def test_insert_slice_rankn_eval_matches_reference_scatter():
    """AC oracle: a rank-3 window at ``(0, P%128, 0)`` (the middle offset a
    runtime member) evaluates to the same tensor as a reference scatter."""
    torch.manual_seed(0)
    dst = torch.randn(1, 16512, 512)
    upd = torch.randn(1, 1, 512)
    p = 640 % 128  # a runtime middle-axis offset
    out = _eval_rankn(dst, upd, (0, p, 0), runtime_axis=1)
    torch.testing.assert_close(out, _ref_scatter(dst, upd, (0, p, 0)))


def test_insert_slice_rankn_eval_all_literal():
    torch.manual_seed(1)
    dst = torch.randn(2, 8, 4)
    upd = torch.randn(1, 3, 4)
    out = _eval_rankn(dst, upd, (1, 2, 0))
    torch.testing.assert_close(out, _ref_scatter(dst, upd, (1, 2, 0)))


def test_insert_slice_rankn_eval_runtime_oob_raises():
    """A runtime offset member that puts the window out of bounds is caught by
    the eval guard (typeinfer cannot see the runtime value)."""
    dst = torch.zeros(1, 8, 4)
    upd = torch.zeros(1, 3, 4)
    with pytest.raises(ValueError, match="out of bounds"):
        _eval_rankn(dst, upd, (0, 6, 0), runtime_axis=1)  # 6 + 3 > 8 on axis 1


def _ref(dst, upd, start):
    out = dst.clone()
    out[start:start + upd.shape[0]] = upd
    return out


@pytest.mark.parametrize(
    "dst,upd,start",
    [
        (torch.zeros(8), torch.tensor([1.0, 2.0, 3.0]), 2),   # interior window
        (torch.arange(6.0), torch.tensor([9.0]), 0),          # single element at 0
        (torch.arange(6.0), torch.tensor([7.0]), 5),          # single element at end
        (torch.zeros(4), torch.arange(1.0, 5.0), 0),          # full overwrite
    ],
    ids=["interior", "elem_start", "elem_end", "full"],
)
def test_insert_slice_eval(dst, upd, start):
    offs = torch.tensor(start, dtype=torch.int32)
    run_eval_case(EvalCase("", _OP, (dst, upd, offs), _ref(dst, upd, start), atol=0.0))


@pytest.mark.parametrize(
    "dst,upd,start",
    [
        (torch.zeros(8), torch.tensor([1.0, 2.0, 3.0]), -1),  # negative start
        (torch.zeros(8), torch.tensor([1.0, 2.0, 3.0]), 6),   # window runs past end
    ],
    ids=["negative_start", "past_end"],
)
def test_insert_slice_eval_out_of_bounds(dst, upd, start):
    """A runtime offset that puts the window out of dst's bounds is rejected by
    the eval guard (the static typeinfer check cannot see a runtime offset)."""
    from dataclasses import replace  # noqa: PLC0415

    from tilefoundry.evaluator import evaluate  # noqa: PLC0415
    from tilefoundry.ir.core import Call, Var  # noqa: PLC0415
    from tilefoundry.ir.hir.function import Function  # noqa: PLC0415
    from tilefoundry.ir.types import TensorType  # noqa: PLC0415
    from tilefoundry.visitor_registry.contexts import TypeInferContext  # noqa: PLC0415
    from tilefoundry.visitor_registry.visitors import TypeInferVisitor  # noqa: PLC0415

    offs = torch.tensor(start, dtype=torch.int32)
    inputs = (dst, upd, offs)
    dtypes = (_F, _F, _I)
    params = tuple(
        Var(type=TensorType(shape=tuple(t.shape), dtype=d, layout=None, storage="gmem"), name=f"x{i}")
        for i, (t, d) in enumerate(zip(inputs, dtypes))
    )
    call = Call(type=params[0].type, target=_OP, args=params)
    result_type = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=result_type)
    fn = Function.build(name="eval_oob", params=params, body=call, return_type=result_type)
    with pytest.raises(ValueError, match="out of bounds"):
        evaluate(fn, *inputs, device="cpu")


# ── single decode step: in-place carry lowering + reshard-owned sync ──────

from tilefoundry import func, module  # noqa: E402
from tilefoundry.dsl import Mesh, Tensor, Topology  # noqa: E402
from tilefoundry.dsl.storage import gmem  # noqa: E402
from tilefoundry.dsl.tf import *  # noqa: E402,F401,F403
from tilefoundry.ir.core.module import Module  # noqa: E402
from tilefoundry.ir.tir.memory.alloc_tensor import AllocTensor  # noqa: E402
from tilefoundry.ir.tir.memory.copy import Copy as TirCopy  # noqa: E402
from tilefoundry.ir.tir.memory.tensor_view import TensorView as TirTensorView  # noqa: E402
from tilefoundry.ir.tir.stmts import (  # noqa: E402
    Evaluate,
    For,
    LetStmt,
    MeshScope,
    Sequential,
)
from tilefoundry.ir.types.shard import Layout as ShardCuteLayout  # noqa: E402
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

    @func(topologies=(Topology("thread", 1),))
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

    rm = tilefoundry.compile(_DecodeStep, target="cuda")
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


def test_decode_step_in_place_carry() -> None:
    """The loop-carried ``acc = insert_slice(acc, …)`` reuses a single carry
    buffer: the buffer is allocated once before the loop, written in place via a
    slice-view Copy inside the loop, with no replacement allocation in the loop
    body (the yielded result aliases the carry buffer)."""
    pf = _lower(_DecodeStep)
    nodes = []
    _walk(pf.body, False, nodes)

    alloc_before = {
        id(var)
        for in_loop, kind, var, val in nodes
        if kind == "let" and not in_loop and isinstance(_op_of(val), AllocTensor)
    }
    alloc_in_loop = {
        id(var)
        for in_loop, kind, var, val in nodes
        if kind == "let" and in_loop and isinstance(_op_of(val), AllocTensor)
    }
    # In-loop slice-view window over a carry buffer (an rmem AllocTensor result,
    # not a kernel-param cache): this is the in-place insert_slice window.
    windows = [
        (var, val.args[0])
        for in_loop, kind, var, val in nodes
        if kind == "let"
        and in_loop
        and isinstance(_op_of(val), TirTensorView)
        and len(val.args) > 1
        and id(val.args[0]) in alloc_before
    ]
    assert windows, "no in-place insert_slice window over a carry buffer found"
    win_var, carry_buf = windows[0]

    # The carry buffer is allocated once before the loop and never re-allocated
    # inside it (single reused buffer, no replacement alloc).
    assert id(carry_buf) in alloc_before
    assert id(carry_buf) not in alloc_in_loop

    # The window is written in place (a Copy whose destination is the window).
    copied_into_window = any(
        kind == "eval"
        and isinstance(_op_of(val), TirCopy)
        and len(val.args) == 2
        and val.args[1] is win_var
        for in_loop, kind, var, val in nodes
    )
    assert copied_into_window, "insert_slice window is not written in place"


@module(entry="xreshard")
class _CrossCtaReshardOutput:
    """A cross-CTA ownership-change reshard (split axis 0 → split axis 1) at the
    output position (the returned value). The reshard-owned grid fence must not
    be bypassed by the output-sink lowering."""

    @func(topologies=(Topology("cta", 2),))
    def xreshard(a: Tensor[(2, _DEC_D), "f32"]) -> Tensor[(2, _DEC_D), "f32"]:
        with Mesh(topology="cta", layout=ShardCuteLayout(shape=(2,), strides=(1,))) as cta:
            g1 = reshard(a, layout=(2 @ cta, _DEC_D), storage=gmem)
            return reshard(g1, layout=(2, _DEC_D @ cta), storage=gmem)


def test_cross_cta_reshard_owned_sync() -> None:
    """An output-position cross-CTA reshard (ownership change) still lowers to
    sync-then-reshard: the grid sync is emitted before the output copy, proving
    the output-sink path routes through the same reshard-owned fence as an
    intermediate reshard. The removed ``_dirty_roots`` heuristic leaves no
    residue."""
    pf = _lower(
        Module(name="m", functions=(_CrossCtaReshardOutput.functions[0],), entry="xreshard")
    )
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

    import inspect  # noqa: PLC0415

    from tilefoundry.passes.transforms import hir_to_tir  # noqa: PLC0415

    assert "_dirty_roots" not in inspect.getsource(hir_to_tir), (
        "the _dirty_roots heuristic must be fully removed"
    )


_DYN_D = 5
_DYN_K = 3


@module(entry="dyn")
class _DynOffset:
    """A loop-carried in-place ``insert_slice`` whose offset is the loop
    induction variable (a rank-0 scalar), writing a length-1 update to a
    different position each iteration."""

    @func(topologies=(Topology("thread", 1),))
    def dyn(base: Tensor[(_DYN_D,), "f32"], v: Tensor[(1,), "f32"]):
        with Mesh(Topology("thread", 1), (1,), ("t",)) as m:
            br = reshard(base, (_DYN_D @ m.t,), "rmem")
            vr = reshard(v, (1 @ m.t,), "rmem")
            acc = full_like(br, 0.0)
            for i in tile(_DYN_K):
                acc = insert_slice(acc, vr, i)
            return reshard(acc, (_DYN_D @ m.t,), "gmem")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_insert_slice_dynamic_offset() -> None:
    """A per-iteration dynamic offset (the loop induction variable) writes the
    update to each position ``0..K-1`` in turn; positions past the loop stay at
    the ``full_like`` init. This exercises non-zero runtime offsets, not just
    offset 0 or a static path."""
    import tilefoundry  # noqa: PLC0415

    rm = tilefoundry.compile(_DynOffset, target="cuda")
    base = torch.randn(_DYN_D, device="cuda")
    v = torch.randn(1, device="cuda")
    out = torch.empty(_DYN_D, device="cuda")
    rm(base, v, out)
    torch.cuda.synchronize()

    exp = torch.zeros(_DYN_D, device="cuda")
    exp[:_DYN_K] = v[0]  # positions 0..K-1 each written with v at offset i
    assert torch.allclose(out, exp, rtol=1e-4, atol=1e-4), (out - exp).abs().max()


# ── rank-N in-place lowering: N-coord window view of the carry buffer ──────

_ND_A, _ND_B, _ND_C, _ND_K = 1, 8, 4, 3


@module(entry="nd_carry")
class _NdCarry:
    """A loop-carried rank-3 in-place ``insert_slice`` writing a length-1 window
    on axis 1 (offset = induction var) each iteration, full on the last axis."""

    @func(topologies=(Topology("thread", 1),))
    def nd_carry(
        base: Tensor[(_ND_A, _ND_B, _ND_C), "f32"],
        v: Tensor[(_ND_A, 1, _ND_C), "f32"],
    ):
        with Mesh(Topology("thread", 1), (1,), ("t",)) as m:
            br = reshard(base, (_ND_A, _ND_B, _ND_C @ m.t), "rmem")
            vr = reshard(v, (_ND_A, 1, _ND_C @ m.t), "rmem")
            acc = full_like(br, 0.0)
            for i in tile(_ND_K):
                acc = insert_slice(acc, vr, (0, i, 0))
            return reshard(acc, (_ND_A, _ND_B, _ND_C @ m.t), "gmem")


def test_insert_slice_rankn_in_place_carry() -> None:
    """The rank-3 in-place ``insert_slice`` lowers to a TensorView window over
    the existing carry buffer using three independent coordinates, written in
    place — no replacement destination allocation in the loop."""
    pf = _lower(_NdCarry)
    nodes = []
    _walk(pf.body, False, nodes)

    alloc_before = {
        id(var) for in_loop, kind, var, val in nodes
        if kind == "let" and not in_loop and isinstance(_op_of(val), AllocTensor)
    }
    alloc_in_loop = {
        id(var) for in_loop, kind, var, val in nodes
        if kind == "let" and in_loop and isinstance(_op_of(val), AllocTensor)
    }
    windows = [
        (var, val) for in_loop, kind, var, val in nodes
        if kind == "let" and in_loop
        and isinstance(_op_of(val), TirTensorView)
        and len(val.args) > 1
        and id(val.args[0]) in alloc_before
    ]
    assert windows, "no in-place rank-N insert_slice window over a carry buffer found"
    win_var, win_val = windows[0]

    # Three independent coordinates: memory + one coord per axis.
    assert len(win_val.args) == 1 + 3, (
        f"expected dst + 3 coords, got {len(win_val.args)} args"
    )
    carry_buf = win_val.args[0]
    assert id(carry_buf) in alloc_before, "window must view a buffer allocated before the loop"
    assert id(carry_buf) not in alloc_in_loop, "window must not view a replacement allocation"

    copied = any(
        kind == "eval"
        and isinstance(_op_of(val), TirCopy)
        and len(val.args) == 2
        and val.args[1] is win_var
        for in_loop, kind, var, val in nodes
    )
    assert copied, "rank-N insert_slice window is not written in place"


# ── rank-N insert_slice end-to-end on GPU: per-axis window + coords ────────
#
# A non-contiguous per-axis window (partial inner axis) written at a dynamic,
# non-zero middle-axis coordinate. This is the numerical gate for the rank-N
# codegen: it fails if the emitter drops trailing coordinates (writes at the
# wrong axis) or flat-collapses the window shape (a partial inner axis is not
# contiguous in the flattened buffer).
_NW_A, _NW_B, _NW_C = 1, 4, 6
_NW_UB, _NW_UC = 2, 3  # window extent on axis 1 / axis 2 (partial: 3 of 6)
_NW_STEPS = 2


@module(entry="nd_window")
class _NdWindow:
    """A loop-carried rank-3 in-place ``insert_slice`` writing a non-trivial,
    non-contiguous window (full axis 0, window 2 on axis 1, partial 3-of-6 on
    axis 2) at the induction variable as the middle-axis tile coordinate."""

    @func(topologies=(Topology("thread", 1),))
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


def _cuda_source(cls) -> str:
    import tilefoundry  # noqa: PLC0415
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    return emit_cuda_module(
        group_functions_by_target(tilefoundry.lower(cls, target="cuda"))["cuda"]
    ).source


def test_insert_slice_rankn_codegen_covers_retained_axes() -> None:
    """Auxiliary source check: the rank-N window tile emits one coordinate and
    one window extent per *retained* per-thread axis (degenerate extent-1 axes
    are elided, so this is ``make_coord(i, 0)`` here) — not a single flat coord /
    flat tile size."""
    import re  # noqa: PLC0415

    src = _cuda_source(_NdWindow)
    tiles = re.findall(r"cute::local_tile\((.*?)\);", src, re.DOTALL)
    assert tiles, "no local_tile emitted for the rank-N insert_slice window"
    nd = [
        t for t in tiles
        if "tilefoundry::local(" in t and re.search(r"make_coord\([^)]*,", t)
    ]
    assert nd, f"no N-D local_tile over a per-thread view found; got: {tiles}"
    call = nd[0]
    coords = [c.strip() for c in re.search(r"make_coord\((.*?)\)", call).group(1).split(",")]
    shape = [s.strip() for s in re.search(r"make_shape\((.*?)\)", call).group(1).split(",")]
    # One coordinate and one window extent per retained axis: coord rank ==
    # window rank, more than one axis (not a flat single coord), and the dynamic
    # middle coord is not dropped to a constant 0.
    assert len(coords) == len(shape) >= 2, f"coords={coords} shape={shape}"
    assert any(c != "0" for c in coords), f"all coordinates collapsed to 0: {coords}"


def _rankn_window_let(mod):
    """The lowered N-D ``insert_slice`` window LetStmt (TensorView, >2 coords)."""
    def walk(node):
        if isinstance(node, Sequential):
            for s in node.body:
                yield from walk(s)
        elif isinstance(node, MeshScope):
            yield from walk(node.body)
        elif isinstance(node, For):
            yield from walk(node.body)
        elif isinstance(node, LetStmt):
            yield node
            yield from walk(node.body)

    for let in walk(_lower(mod).body):
        v = let.value
        if isinstance(_op_of(v), TirTensorView) and len(v.args) > 2:
            return let
    raise AssertionError("no rank-N TensorView window LetStmt found")


@pytest.mark.parametrize("delta", [1, -1], ids=["extra_coord", "missing_coord"])
def test_insert_slice_rankn_codegen_coord_count_fail_closed(delta) -> None:
    """The N-D window emitter fail-closes when the coordinate count does not
    match the layout-derived destination rank — a missing or extra coordinate
    raises rather than silently ignoring or fabricating an axis."""
    import dataclasses  # noqa: PLC0415

    import tilefoundry.codegen.cuda  # noqa: F401,PLC0415 — emitter autodiscovery
    from tilefoundry.codegen.cuda.context import CodegenContext  # noqa: PLC0415
    from tilefoundry.codegen.cuda.tir.memory.tensor_view import _emit  # noqa: PLC0415

    let = _rankn_window_let(_NdWindow)
    coords = let.value.args[1:]
    if delta > 0:
        bad = let.value.args + (coords[-1],)
    else:
        bad = let.value.args[:-1]
    bad_let = dataclasses.replace(let, value=dataclasses.replace(let.value, args=bad))
    with pytest.raises(ValueError, match="offsets"):
        _emit(bad_let, CodegenContext())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_insert_slice_rankn_gpu_oracle() -> None:
    """The rank-N in-place ``insert_slice`` runs on GPU and matches a torch
    scatter reference: a non-contiguous window at a dynamic, non-zero
    middle-axis coordinate."""
    import tilefoundry  # noqa: PLC0415

    rm = tilefoundry.compile(_NdWindow, target="cuda")
    base = torch.randn(_NW_A, _NW_B, _NW_C, device="cuda")
    v = torch.randn(_NW_A, _NW_UB, _NW_UC, device="cuda")
    out = torch.empty(_NW_A, _NW_B, _NW_C, device="cuda")
    rm(base, v, out)
    torch.cuda.synchronize()

    exp = torch.zeros(_NW_A, _NW_B, _NW_C, device="cuda")
    for i in range(_NW_STEPS):
        exp[:, _NW_UB * i : _NW_UB * i + _NW_UB, 0:_NW_UC] = v
    assert torch.allclose(out, exp, rtol=1e-4, atol=1e-4), (out - exp).abs().max()
