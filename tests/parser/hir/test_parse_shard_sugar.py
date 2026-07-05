"""Shard-layout sugar parse tests.

Each test owns its mesh, writes the compact ``ShardLayout`` sugar in a nested
``@func`` param annotation, and asserts the parsed ``TensorType`` equals a
hand-written expected value (or that an invalid form raises). Covers inline
``Split``, the ``{...}`` ``Partial`` value-state set, default ``Broadcast``,
multi-mesh-axis split, explicit strides, the single-axis ``int @ mesh``
shorthand, and that sugar source prints to valid Python.
"""

from __future__ import annotations

import ast

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- binds bare op names (reshard, ...)
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import (
    Layout,
    Mesh,
    P,
    ShardLayout,
    Topology,
)
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, Split
from tilefoundry.parser.sugar import parse_shard_layout_sugar
from tests.fixtures.demo_ir import build_demo

# ── inline Split + default Broadcast ────────────────────────────────────────

_M_SPLIT = Mesh(
    Topology("gpu", 8192),
    Layout((32, 2, 8, 32), (2048, 1024, 32, 1)),
    names=("cluster", "cta", "warp", "lane"),
)


def test_split_inline_and_default_broadcast() -> None:
    """``dim @ mesh.axis`` binds a Split on that cute axis; mesh axes named in
    no Split default to Broadcast; cute strides auto-fill C-order."""

    @func
    def _f(
        a: Tensor[(32, 128), bf16, (32 @ _M_SPLIT.cluster, 2 @ _M_SPLIT.cta, 64), "smem"],
    ) -> Tensor[(32, 128), "f32"]:
        return a

    assert _f.params[0].type == TensorType(
        shape=(32, 128),
        dtype=DType.bf16,
        storage=StorageKind.SMEM,
        layout=ShardLayout(
            layout=Layout((32, 2, 64), (128, 64, 1)),
            attrs=(Split(0), Split(1), Broadcast(), Broadcast()),
            mesh=_M_SPLIT,
        ),
    )


# ── Partial value-state set ──────────────────────────────────────────────────

_M_PARTIAL = Mesh(
    Topology("gpu", 8192),
    Layout((32, 2, 8, 32), (2048, 1024, 32, 1)),
    names=("cluster", "cta", "warp", "lane"),
)


def test_partial_brace_value_state() -> None:
    """The optional final ``{mesh.axis @ P("reduction")}`` set carries a
    mesh-axis Partial value state; the layout tuple holds only Split placement,
    and unnamed axes stay Broadcast."""

    @func
    def _f(
        a: Tensor[
            (64, 128), bf16, ((32 @ _M_PARTIAL.cluster, 64), {_M_PARTIAL.warp @ P("sum")}), "smem"
        ],
    ) -> Tensor[(64, 128), "f32"]:
        return a

    assert _f.params[0].type == TensorType(
        shape=(64, 128),
        dtype=DType.bf16,
        storage=StorageKind.SMEM,
        layout=ShardLayout(
            layout=Layout((32, 64), (64, 1)),
            attrs=(Split(0), Broadcast(), Partial("sum"), Broadcast()),
            mesh=_M_PARTIAL,
        ),
    )


# ── mixed Split + Partial + default Broadcast on one mesh ────────────────────

_M_MIXED = Mesh(
    Topology("thread", 4 * 2 * 16), Layout((4, 2, 16), (32, 16, 1)), names=("l", "g", "t")
)


def test_mixed_split_partial_and_default_broadcast() -> None:
    """``l`` splits dim 0, ``t`` is a Partial value state, and the unnamed ``g``
    defaults to Broadcast."""

    @func
    def _f(
        a: Tensor[(4, 64), "f32", ((4 @ _M_MIXED.l, 64), {_M_MIXED.t @ P("sum")}), "smem"],
    ) -> Tensor[(4, 64), "f32"]:
        return a

    assert _f.params[0].type == TensorType(
        shape=(4, 64),
        dtype=DType.f32,
        storage=StorageKind.SMEM,
        layout=ShardLayout(
            layout=Layout((4, 64), (64, 1)),
            attrs=(Split(0), Broadcast(), Partial("sum")),
            mesh=_M_MIXED,
        ),
    )


# ── multi-mesh-axis split: ``dim @ (mesh.axis, ...)`` ────────────────────────

_M_MULTI = Mesh(Topology("thread", 6 * 32), Layout((6, 32), (32, 1)), names=("w", "t"))


def test_multi_axis_split_factorises_with_remainder() -> None:
    """``1536 @ (w, t)`` factorises the dim into the mesh extents (6, 32) plus a
    remainder (8), each extent bound as a Split; the leading unit axis is kept."""

    @func
    def _f(
        a: Tensor[(1, 1536), "f32", (1, 1536 @ (_M_MULTI.w, _M_MULTI.t)), "smem"],
    ) -> Tensor[(1, 1536), "f32"]:
        return a

    assert _f.params[0].type == TensorType(
        shape=(1, 1536),
        dtype=DType.f32,
        storage=StorageKind.SMEM,
        layout=ShardLayout(
            layout=Layout((1, 6, 32, 8), (1536, 256, 8, 1)),
            attrs=(Split(1), Split(2)),
            mesh=_M_MULTI,
        ),
    )


def test_multi_axis_split_factorises_exact_plus_remainder() -> None:
    """``384 @ (w, t)`` factorises into (6, 32) plus the remainder 2; the
    single-axis canonicalization path does not apply to a multi-axis split."""

    @func
    def _f(
        a: Tensor[(384,), "f32", (384 @ (_M_MULTI.w, _M_MULTI.t),), "smem"],
    ) -> Tensor[(384,), "f32"]:
        return a

    assert _f.params[0].type == TensorType(
        shape=(384,),
        dtype=DType.f32,
        storage=StorageKind.SMEM,
        layout=ShardLayout(
            layout=Layout((6, 32, 2), (64, 2, 1)),
            attrs=(Split(0), Split(1)),
            mesh=_M_MULTI,
        ),
    )


def test_multi_axis_split_not_divisible_raises() -> None:
    """A dim must be divisible by the product of the mesh extents."""

    with pytest.raises(ValueError, match="not divisible"):

        @func
        def _bad(
            a: Tensor[(1, 100), "f32", (1, 100 @ (_M_MULTI.w, _M_MULTI.t)), "smem"],
        ) -> Tensor[(1, 100), "f32"]:
            return a


# ── explicit strides: ``((dims), (strides))`` ────────────────────────────────

_M_STRIDED = Mesh(Topology("thread", 4 * 32), Layout((4, 32), (32, 1)), names=("y", "t"))


def test_explicit_strides_skip_single_axis_canonicalization() -> None:
    """The ``((dims), (strides))`` form preserves user-supplied dims and
    strides; the explicit-strides path does not trigger single-axis
    canonicalization."""

    @func
    def _f(
        a: Tensor[(12, 4), "f32", ((12 @ _M_STRIDED.y, 4), (4, 1)), "smem"],
    ) -> Tensor[(12, 4), "f32"]:
        return a

    assert _f.params[0].type == TensorType(
        shape=(12, 4),
        dtype=DType.f32,
        storage=StorageKind.SMEM,
        layout=ShardLayout(
            layout=Layout((12, 4), (4, 1)),
            attrs=(Split(0), Broadcast()),
            mesh=_M_STRIDED,
        ),
    )


# ── single-axis ``int @ mesh`` shorthand ─────────────────────────────────────

_M_CTA = Mesh(Topology("cta", 128), Layout((128,), (1,)), names=("cta",))


def test_int_at_single_axis_mesh_canonicalises() -> None:
    """On a single-axis mesh, ``8192 @ cta`` (extent 128) canonicalises into
    ``(128, 64)`` with the mesh axis bound as a Split on the new cute axis."""

    @func
    def _f(
        a: Tensor[(1, 8192), "f32", (1, 8192 @ _M_CTA), "smem"],
    ) -> Tensor[(1, 8192), "f32"]:
        return a

    assert _f.params[0].type == TensorType(
        shape=(1, 8192),
        dtype=DType.f32,
        storage=StorageKind.SMEM,
        layout=ShardLayout(
            layout=Layout((1, 128, 64), (8192, 64, 1)),
            attrs=(Split(1),),
            mesh=_M_CTA,
        ),
    )


_M_MULTI_REJECT = Mesh(
    Topology("thread", 4 * 2 * 16), Layout((4, 2, 16), (32, 16, 1)), names=("l", "g", "t")
)


def test_int_at_mesh_rejects_multi_axis_mesh() -> None:
    """The bare ``int @ mesh`` shorthand requires a single-axis mesh; a
    multi-axis mesh still needs an explicit ``mesh.axis`` reference."""

    with pytest.raises(ValueError, match="single-axis mesh"):

        @func
        def _bad(
            a: Tensor[(64,), "f32", (64 @ _M_MULTI_REJECT,), "smem"],
        ) -> Tensor[(64,), "f32"]:
            return a


# ── invalid value-state forms ────────────────────────────────────────────────

_M_VALUE_STATE = Mesh(
    Topology("thread", 4 * 2 * 16), Layout((4, 2, 16), (32, 16, 1)), names=("l", "g", "t")
)


def test_value_state_set_must_be_final_outer_item() -> None:
    """The ``{...}`` value-state set is valid only as the last outer item; a
    stride tuple after it is rejected."""

    with pytest.raises(ValueError, match="last outer item"):

        @func
        def _bad(
            a: Tensor[
                (4, 64),
                "f32",
                ((4 @ _M_VALUE_STATE.l, 64), {_M_VALUE_STATE.t @ P("sum")}, (64, 1)),
                "smem",
            ],
        ) -> Tensor[(4, 64), "f32"]:
            return a


def test_value_state_p_requires_reduction_arg() -> None:
    """``P(...)`` in the value-state set requires its reduction argument; bare
    ``P()`` is rejected (the surface is ``mesh.axis @ P("reduction")``)."""

    with pytest.raises(ValueError, match="reduction argument"):

        @func
        def _bad(
            a: Tensor[(4, 64), "f32", ((4 @ _M_VALUE_STATE.l, 64), {_M_VALUE_STATE.t @ P()}), "smem"],
        ) -> Tensor[(4, 64), "f32"]:
            return a


# ── undefined mesh / unknown axis ────────────────────────────────────────────

_M_KNOWN = Mesh(
    Topology("gpu", 8192), Layout((32, 2), (2, 1)), names=("cluster", "cta")
)


def test_sugar_undefined_mesh_raises() -> None:
    """Sugar that references a name bound to no mesh raises."""

    with pytest.raises(ValueError, match="undefined mesh"):

        @func
        def _bad(
            a: Tensor[(32, 128), bf16, (32 @ undefined.cluster, 64), "smem"],  # noqa: F821
        ) -> Tensor[(32, 128), "f32"]:
            return a


def test_sugar_unknown_axis_raises() -> None:
    """Sugar that references an axis the mesh does not have raises."""

    with pytest.raises(ValueError, match="has no axis named"):

        @func
        def _bad(
            a: Tensor[(32, 128), bf16, (32 @ _M_KNOWN.nonexistent, 64), "smem"],
        ) -> Tensor[(32, 128), "f32"]:
            return a


# ── ``@func`` body with a sugar param + printing valid Python ────────────────

_M_BODY = Mesh(
    Topology("gpu", 8192),
    Layout((32, 2, 8, 32), (2048, 1024, 32, 1)),
    names=("cluster", "cta", "warp", "lane"),
)


def test_func_body_with_sugar_param_parses_and_reshards() -> None:
    """A ``@func`` whose param carries sugar parses to the hand-written type and
    its ``reshard`` body op fires."""

    @func
    def _demo(
        a: Tensor[(32, 1536), "f32", (32 @ _M_BODY.cluster, 1536), "smem"],
    ) -> Tensor[(32, 1536), "f32"]:
        return reshard(
            a,
            layout=ShardLayout(
                layout=Layout((32, 1536), (1536, 1)),
                attrs=(),
                mesh=Mesh(Topology("cta", 128), Layout((128,), (1,))),
            ),
        )

    fn = _demo
    assert fn.name == "_demo"
    assert fn.params[0].type == TensorType(
        shape=(32, 1536),
        dtype=DType.f32,
        storage=StorageKind.SMEM,
        layout=ShardLayout(
            layout=Layout((32, 1536), (1536, 1)),
            attrs=(Split(0), Broadcast(), Broadcast(), Broadcast()),
            mesh=_M_BODY,
        ),
    )
    assert isinstance(fn.body, Call) and isinstance(fn.body.target, Reshard)


def test_sugar_source_prints_to_valid_python() -> None:
    """A function with a sugar param annotation prints to valid Python source,
    and the parsed param keeps its shape/dtype/storage."""

    @func
    def _demo(
        a: Tensor[(32, 1536), "f32", (32 @ _M_BODY.cluster, 1536), "smem"],
    ) -> Tensor[(32, 1536), "f32"]:
        return a

    fn = _demo
    src = as_script(fn, module="M")
    compile(src, "<test>", "exec")  # printed output is valid Python
    p = fn.params[0]
    assert p.type.shape == (32, 1536)
    assert p.type.dtype == DType.f32
    assert p.type.storage == StorageKind.SMEM


def test_printer_falls_back_to_verbose_when_mesh_has_no_names() -> None:
    """A mesh without ``names=`` cannot use ``@`` sugar; the printer emits the
    verbose ``ShardLayout(...)`` form instead."""
    fn, _, _ = build_demo()
    src = as_script(fn)
    assert "@" not in src.split("@func")[1].split("def ")[0]
    assert "ShardLayout(" in src


# ── dynamic (DimVar) / closure-Name axis extents ─────────────────────────────

_S_DYN = DimVar("seq_len", 1, 4)


def test_reshard_sugar_accepts_dynamic_bare_and_closure_name_axis() -> None:
    """A reshard layout sugar may carry a dynamic ``DimVar`` bare axis (``S``)
    and a closure-resolved Name split extent (``_HQ``). The split axis is
    canonicalised against the mesh extent; the dynamic axis rides through as a
    Broadcast dim. The reshard's logical result keeps the un-factorised shape
    and strides defer to typeinfer."""
    _HQ, _D = 32, 128

    @func(topologies=(Topology("cta", 8),))
    def _f(
        q: Tensor[(1, _S_DYN, _HQ, _D), "bf16"],
    ) -> Tensor[(1, _S_DYN, _HQ, _D), "bf16"]:
        with Mesh(topology="cta", layout=Layout((8,), (1,))) as cta:
            return reshard(q, layout=(1, _S_DYN, _HQ @ cta, _D))  # noqa: F821

    body = _f.body
    assert isinstance(body, Call) and isinstance(body.target, Reshard)
    # logical result shape is the un-factorised (1, S, 32, 128)
    assert body.type.shape == (1, _S_DYN, _HQ, _D)
    # the head axis is Split across the cta mesh
    assert any(isinstance(a, Split) for a in body.target.layout.attrs)
    # the authored sugar leaves strides un-materialised (deferred to typeinfer)
    assert body.target.layout.layout.strides is None


def test_reshard_sugar_rejects_dynamic_split_axis() -> None:
    """A bare axis may be dynamic, but a *split* axis (``dim @ mesh.axis``)
    participates in canonicalisation and must resolve to a static int — a
    dynamic ``DimVar`` split extent is rejected."""
    cta = Mesh(Topology("cta", 8), Layout((8,), (1,)), names=("cta",))
    node = ast.parse("(1, S @ cta, 32, 128)", mode="eval").body
    with pytest.raises(ValueError, match="static int"):
        parse_shard_layout_sugar(
            node, lambda n: cta if n == "cta" else None, closure={"S": _S_DYN}
        )


# ── static-int dims in mesh / split sugar: closure resolution + diagnostics ───
# A static-extent (mesh-shape dim or split extent) accepts an int literal or a
# closure/global int; a bool or dynamic (DimVar) value is rejected with a
# static-int diagnostic (and the sugar error must surface, not be swallowed into
# ``'Mesh' object has no attribute ...``). All cases below use the string-
# topology sugar `Mesh(topology="thread", ...)` referencing the @func-declared
# topology rather than reconstructing a ``Topology(...)`` in the body.

_MESH_DIM_W = DimVar("W", 1, 8)
_SPLIT_EXTENT_K = DimVar("K", 32, 256)


def _closure_mesh_dims_fn(warps, lanes):
    """A mesh-shape sugar (`layout=(warps, lanes)`) whose dims come from the
    enclosing closure."""

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(warps, lanes), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def _closure_split_extent_fn(k_tile):
    """A split extent (`k_tile @ (m.w, m.t)`) taken from the enclosing closure."""

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, k_tile @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def _literal_reshard_fn():
    """All-literal reference form: `layout=(4, 32)` mesh dims and a `128 @
    (m.w, m.t)` split extent. Both closure builders above must print to this."""

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


@pytest.mark.parametrize(
    "closure_call",
    [
        lambda: _closure_mesh_dims_fn(4, 32),
        lambda: _closure_split_extent_fn(128),
    ],
    ids=["mesh-dims", "split-extent"],
)
def test_closure_int_resolves_like_literal(closure_call) -> None:
    """A closure/global int in a static-extent position (mesh dim or split
    extent) resolves identically to the integer literal — the parser must not
    reject the ``ast.Name``; the closure form prints back to the literal form."""
    assert as_script(closure_call()) == as_script(_literal_reshard_fn())


def _dimvar_mesh_dim_fn():
    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(_MESH_DIM_W, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def _bool_mesh_dim_fn():
    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(True, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def _dimvar_split_extent_fn():
    @func(topologies=(Topology("cta", 8), Topology("thread", 128)))
    def _f(x: Tensor[(8, _SPLIT_EXTENT_K), "bf16"]) -> Tensor[(8, 1), "bf16"]:
        with Mesh(topology="thread", layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (8, _SPLIT_EXTENT_K @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (8, 1), "gmem")  # noqa: F821

    return _f


def _bool_split_extent_single_axis_fn():
    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, True @ m.w), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def _bool_split_extent_multi_axis_fn():
    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, True @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


@pytest.mark.parametrize(
    "build",
    [
        _dimvar_mesh_dim_fn,
        _bool_mesh_dim_fn,
        _dimvar_split_extent_fn,
        _bool_split_extent_single_axis_fn,
        _bool_split_extent_multi_axis_fn,
    ],
    ids=[
        "dimvar-mesh-dim",
        "bool-mesh-dim",
        "dimvar-split-extent",
        "bool-split-single-axis",
        "bool-split-multi-axis",
    ],
)
def test_static_extent_position_rejects_non_static_int(build) -> None:
    """A dynamic (``DimVar``) or ``bool`` value in a static-extent position (mesh
    dim or split extent) is rejected with a clear static-int diagnostic; the
    sugar error must surface rather than be swallowed into a generic
    ``'Mesh' object has no attribute ...`` attribute error."""
    with pytest.raises(ValueError, match="static int"):
        build()
