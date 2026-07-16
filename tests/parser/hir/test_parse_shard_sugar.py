"""Shard-layout sugar parse tests.

Unified model: each scenario is a named ``build_*_func`` / ``build_*_case``
builder carrying a docstring that describes the DSL scene, and a ``test_*`` that
runs it through a shared assertion helper (parsed ``TensorType`` equality, or a
raised diagnostic). Covers inline ``Split``, the ``{...}`` ``Partial``
value-state set, default ``Broadcast``, multi-mesh-axis split, explicit strides,
the single-axis ``int @ mesh`` shorthand, closure/static-int dim resolution, and
that sugar source prints to valid Python.
"""

from __future__ import annotations

import ast
from typing import Any, Callable

import pytest

from tests.models.demo.demo_ir import build_demo
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

# ── shared assertion helpers ─────────────────────────────────────────────────


def assert_param_type(build_func: Callable[[], Any], expected: TensorType) -> None:
    """Build the DSL ``@func`` and assert its first parameter's parsed type."""
    fn = build_func()
    assert fn.params[0].type == expected


def assert_build_raises(build_func: Callable[[], Any], match: str) -> None:
    """Assert that building the DSL ``@func`` raises a ``ValueError`` (parse
    happens at ``@func`` decoration, so calling the builder triggers it)."""
    with pytest.raises(ValueError, match=match):
        build_func()


def assert_parse_raises(build_case: Callable[[], Callable[[], Any]], match: str) -> None:
    """Assert that running a direct ``parse_shard_layout_sugar`` case (returned as
    a zero-arg thunk by *build_case*) raises a ``ValueError``."""
    do_parse = build_case()
    with pytest.raises(ValueError, match=match):
        do_parse()


# ── inline Split + default Broadcast ────────────────────────────────────────

_M_SPLIT = Mesh(
    Topology("gpu", 8192),
    Layout((32, 2, 8, 32), (2048, 1024, 32, 1)),
    names=("cluster", "cta", "warp", "lane"),
)


def build_split_inline_and_default_broadcast_func():
    """``dim @ mesh.axis`` binds a Split on that layout axis; mesh axes named in
    no Split default to Broadcast; layout strides auto-fill C-order."""

    @func
    def _f(
        a: Tensor[(32, 128), bf16, (32 @ _M_SPLIT.cluster, 2 @ _M_SPLIT.cta, 64), "smem"],
    ) -> Tensor[(32, 128), "f32"]:
        return a

    return _f


def test_split_inline_and_default_broadcast() -> None:
    assert_param_type(
        build_split_inline_and_default_broadcast_func,
        TensorType(
            shape=(32, 128),
            dtype=DType.bf16,
            storage=StorageKind.SMEM,
            layout=ShardLayout(
                layout=Layout((32, 2, 64), (128, 64, 1)),
                attrs=(Split(0), Split(1), Broadcast(), Broadcast()),
                mesh=_M_SPLIT,
            ),
        ),
    )


# ── Partial value-state set ──────────────────────────────────────────────────

_M_PARTIAL = Mesh(
    Topology("gpu", 8192),
    Layout((32, 2, 8, 32), (2048, 1024, 32, 1)),
    names=("cluster", "cta", "warp", "lane"),
)


def build_partial_brace_value_state_func():
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

    return _f


def test_partial_brace_value_state() -> None:
    assert_param_type(
        build_partial_brace_value_state_func,
        TensorType(
            shape=(64, 128),
            dtype=DType.bf16,
            storage=StorageKind.SMEM,
            layout=ShardLayout(
                layout=Layout((32, 64), (64, 1)),
                attrs=(Split(0), Broadcast(), Partial("sum"), Broadcast()),
                mesh=_M_PARTIAL,
            ),
        ),
    )


# ── mixed Split + Partial + default Broadcast on one mesh ────────────────────

_M_MIXED = Mesh(
    Topology("thread", 4 * 2 * 16), Layout((4, 2, 16), (32, 16, 1)), names=("l", "g", "t")
)


def build_mixed_split_partial_and_default_broadcast_func():
    """``l`` splits dim 0, ``t`` is a Partial value state, and the unnamed ``g``
    defaults to Broadcast."""

    @func
    def _f(
        a: Tensor[(4, 64), "f32", ((4 @ _M_MIXED.l, 64), {_M_MIXED.t @ P("sum")}), "smem"],
    ) -> Tensor[(4, 64), "f32"]:
        return a

    return _f


def test_mixed_split_partial_and_default_broadcast() -> None:
    assert_param_type(
        build_mixed_split_partial_and_default_broadcast_func,
        TensorType(
            shape=(4, 64),
            dtype=DType.f32,
            storage=StorageKind.SMEM,
            layout=ShardLayout(
                layout=Layout((4, 64), (64, 1)),
                attrs=(Split(0), Broadcast(), Partial("sum")),
                mesh=_M_MIXED,
            ),
        ),
    )


# ── multi-mesh-axis split: ``dim @ (mesh.axis, ...)`` ────────────────────────

_M_MULTI = Mesh(Topology("thread", 6 * 32), Layout((6, 32), (32, 1)), names=("w", "t"))


def build_multi_axis_split_with_remainder_func():
    """``1536 @ (w, t)`` factorises the dim into the mesh extents (6, 32) plus a
    remainder (8), each extent bound as a Split; the leading unit axis is kept."""

    @func
    def _f(
        a: Tensor[(1, 1536), "f32", (1, 1536 @ (_M_MULTI.w, _M_MULTI.t)), "smem"],
    ) -> Tensor[(1, 1536), "f32"]:
        return a

    return _f


def test_multi_axis_split_factorises_with_remainder() -> None:
    assert_param_type(
        build_multi_axis_split_with_remainder_func,
        TensorType(
            shape=(1, 1536),
            dtype=DType.f32,
            storage=StorageKind.SMEM,
            layout=ShardLayout(
                layout=Layout((1, 6, 32, 8), (1536, 256, 8, 1)),
                attrs=(Split(1), Split(2)),
                mesh=_M_MULTI,
            ),
        ),
    )


def build_multi_axis_split_exact_plus_remainder_func():
    """``384 @ (w, t)`` factorises into (6, 32) plus the remainder 2; the
    single-axis canonicalization path does not apply to a multi-axis split."""

    @func
    def _f(
        a: Tensor[(384,), "f32", (384 @ (_M_MULTI.w, _M_MULTI.t),), "smem"],
    ) -> Tensor[(384,), "f32"]:
        return a

    return _f


def test_multi_axis_split_factorises_exact_plus_remainder() -> None:
    assert_param_type(
        build_multi_axis_split_exact_plus_remainder_func,
        TensorType(
            shape=(384,),
            dtype=DType.f32,
            storage=StorageKind.SMEM,
            layout=ShardLayout(
                layout=Layout((6, 32, 2), (64, 2, 1)),
                attrs=(Split(0), Split(1)),
                mesh=_M_MULTI,
            ),
        ),
    )


def build_multi_axis_split_not_divisible_func():
    """A dim must be divisible by the product of the mesh extents; ``100 @
    (w, t)`` (product 192) is rejected."""

    @func
    def _bad(
        a: Tensor[(1, 100), "f32", (1, 100 @ (_M_MULTI.w, _M_MULTI.t)), "smem"],
    ) -> Tensor[(1, 100), "f32"]:
        return a

    return _bad


def test_multi_axis_split_not_divisible_raises() -> None:
    assert_build_raises(build_multi_axis_split_not_divisible_func, match="not divisible")


# ── explicit strides: ``((dims), (strides))`` ────────────────────────────────

_M_STRIDED = Mesh(Topology("thread", 4 * 32), Layout((4, 32), (32, 1)), names=("y", "t"))


def build_explicit_strides_func():
    """The ``((dims), (strides))`` form preserves user-supplied dims and strides;
    the explicit-strides path does not trigger single-axis canonicalization."""

    @func
    def _f(
        a: Tensor[(12, 4), "f32", ((12 @ _M_STRIDED.y, 4), (4, 1)), "smem"],
    ) -> Tensor[(12, 4), "f32"]:
        return a

    return _f


def test_explicit_strides_skip_single_axis_canonicalization() -> None:
    assert_param_type(
        build_explicit_strides_func,
        TensorType(
            shape=(12, 4),
            dtype=DType.f32,
            storage=StorageKind.SMEM,
            layout=ShardLayout(
                layout=Layout((12, 4), (4, 1)),
                attrs=(Split(0), Broadcast()),
                mesh=_M_STRIDED,
            ),
        ),
    )


# ── single-axis ``int @ mesh`` shorthand ─────────────────────────────────────

_M_CTA = Mesh(Topology("cta", 128), Layout((128,), (1,)), names=("cta",))


def build_int_at_single_axis_mesh_func():
    """On a single-axis mesh, ``8192 @ cta`` (extent 128) canonicalises into
    ``(128, 64)`` with the mesh axis bound as a Split on the new layout axis."""

    @func
    def _f(
        a: Tensor[(1, 8192), "f32", (1, 8192 @ _M_CTA), "smem"],
    ) -> Tensor[(1, 8192), "f32"]:
        return a

    return _f


def test_int_at_single_axis_mesh_canonicalises() -> None:
    assert_param_type(
        build_int_at_single_axis_mesh_func,
        TensorType(
            shape=(1, 8192),
            dtype=DType.f32,
            storage=StorageKind.SMEM,
            layout=ShardLayout(
                layout=Layout((1, 128, 64), (8192, 64, 1)),
                attrs=(Split(1),),
                mesh=_M_CTA,
            ),
        ),
    )


_M_MULTI_REJECT = Mesh(
    Topology("thread", 4 * 2 * 16), Layout((4, 2, 16), (32, 16, 1)), names=("l", "g", "t")
)


def build_int_at_multi_axis_mesh_func():
    """The bare ``int @ mesh`` shorthand requires a single-axis mesh; a
    multi-axis mesh still needs an explicit ``mesh.axis`` reference."""

    @func
    def _bad(
        a: Tensor[(64,), "f32", (64 @ _M_MULTI_REJECT,), "smem"],
    ) -> Tensor[(64,), "f32"]:
        return a

    return _bad


def test_int_at_mesh_rejects_multi_axis_mesh() -> None:
    assert_build_raises(build_int_at_multi_axis_mesh_func, match="single-axis mesh")


# ── invalid value-state forms ────────────────────────────────────────────────

_M_VALUE_STATE = Mesh(
    Topology("thread", 4 * 2 * 16), Layout((4, 2, 16), (32, 16, 1)), names=("l", "g", "t")
)


def build_value_state_not_final_func():
    """The ``{...}`` value-state set is valid only as the last outer item; a
    stride tuple after it is rejected."""

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

    return _bad


def test_value_state_set_must_be_final_outer_item() -> None:
    assert_build_raises(build_value_state_not_final_func, match="last outer item")


def build_value_state_bare_p_func():
    """``P(...)`` in the value-state set requires its reduction argument; bare
    ``P()`` is rejected (the surface is ``mesh.axis @ P("reduction")``)."""

    @func
    def _bad(
        a: Tensor[(4, 64), "f32", ((4 @ _M_VALUE_STATE.l, 64), {_M_VALUE_STATE.t @ P()}), "smem"],
    ) -> Tensor[(4, 64), "f32"]:
        return a

    return _bad


def test_value_state_p_requires_reduction_arg() -> None:
    assert_build_raises(build_value_state_bare_p_func, match="reduction argument")


# ── undefined mesh / unknown axis ────────────────────────────────────────────

_M_KNOWN = Mesh(
    Topology("gpu", 8192), Layout((32, 2), (2, 1)), names=("cluster", "cta")
)


def build_undefined_mesh_func():
    """Sugar that references a name bound to no mesh is rejected."""

    @func
    def _bad(
        a: Tensor[(32, 128), bf16, (32 @ undefined.cluster, 64), "smem"],  # noqa: F821
    ) -> Tensor[(32, 128), "f32"]:
        return a

    return _bad


def test_sugar_undefined_mesh_raises() -> None:
    assert_build_raises(build_undefined_mesh_func, match="undefined mesh")


def build_unknown_axis_func():
    """Sugar that references an axis the mesh does not have is rejected."""

    @func
    def _bad(
        a: Tensor[(32, 128), bf16, (32 @ _M_KNOWN.nonexistent, 64), "smem"],
    ) -> Tensor[(32, 128), "f32"]:
        return a

    return _bad


def test_sugar_unknown_axis_raises() -> None:
    assert_build_raises(build_unknown_axis_func, match="has no axis named")


# ── ``@func`` body with a sugar param + printing valid Python ────────────────

_M_BODY = Mesh(
    Topology("gpu", 8192),
    Layout((32, 2, 8, 32), (2048, 1024, 32, 1)),
    names=("cluster", "cta", "warp", "lane"),
)


def build_body_sugar_param_reshard_func():
    """A ``@func`` whose param carries sugar parses to the hand-written type and
    its ``reshard`` body op fires."""

    @func
    def _demo(
        a: Tensor[(32, 1536), "f32", (32 @ _M_BODY.cluster, 1536), "smem"],
    ) -> Tensor[(32, 1536), "f32"]:
        return reshard(  # noqa: F821
            a,
            layout=ShardLayout(
                layout=Layout((32, 1536), (1536, 1)),
                attrs=(),
                mesh=Mesh(Topology("cta", 128), Layout((128,), (1,))),
            ),
        )

    return _demo


def test_func_body_with_sugar_param_parses_and_reshards() -> None:
    fn = build_body_sugar_param_reshard_func()
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


def build_sugar_param_print_func():
    """A function with a sugar param annotation prints to valid Python source."""

    @func
    def _demo(
        a: Tensor[(32, 1536), "f32", (32 @ _M_BODY.cluster, 1536), "smem"],
    ) -> Tensor[(32, 1536), "f32"]:
        return a

    return _demo


def test_sugar_source_prints_to_valid_python() -> None:
    fn = build_sugar_param_print_func()
    src = as_script(fn, module="M")
    compile(src, "<test>", "exec")  # printed output is valid Python
    p = fn.params[0]
    assert p.type.shape == (32, 1536)
    assert p.type.dtype == DType.f32
    assert p.type.storage == StorageKind.SMEM


def build_no_names_mesh_demo_func():
    """A mesh without ``names=`` cannot use ``@`` sugar; the printer must emit the
    verbose ``ShardLayout(...)`` form instead."""
    fn, _, _ = build_demo()
    return fn


def test_printer_falls_back_to_verbose_when_mesh_has_no_names() -> None:
    fn = build_no_names_mesh_demo_func()
    src = as_script(fn)
    assert "@" not in src.split("@func")[1].split("def ")[0]
    assert "ShardLayout(" in src


# ── dynamic (DimVar) / closure-Name axis extents ─────────────────────────────

_S_DYN = DimVar("seq_len", 1, 4)


def build_dynamic_bare_and_closure_split_func():
    """A reshard layout sugar may carry a dynamic ``DimVar`` bare axis (``S``)
    and a closure-resolved Name split extent (``_HQ``). The split axis is
    canonicalised against the mesh extent; the dynamic axis rides through as a
    Broadcast dim. The reshard's logical result keeps the un-factorised shape and
    strides defer to typeinfer."""
    _HQ, _D = 32, 128

    @func(topologies=(Topology("cta", 8),))
    def _f(
        q: Tensor[(1, _S_DYN, _HQ, _D), "bf16"],
    ) -> Tensor[(1, _S_DYN, _HQ, _D), "bf16"]:
        with Mesh(topology="cta", layout=Layout((8,), (1,))) as cta:
            return reshard(q, layout=(1, _S_DYN, _HQ @ cta, _D))  # noqa: F821

    return _f


def test_reshard_sugar_accepts_dynamic_bare_and_closure_name_axis() -> None:
    fn = build_dynamic_bare_and_closure_split_func()
    body = fn.body
    assert isinstance(body, Call) and isinstance(body.target, Reshard)
    # logical result shape is the un-factorised (1, S, 32, 128)
    assert body.type.shape == (1, _S_DYN, 32, 128)
    # the head axis is Split across the cta mesh
    assert any(isinstance(a, Split) for a in body.target.layout.attrs)
    # the authored sugar leaves strides un-materialised (deferred to typeinfer)
    assert body.target.layout.layout.strides is None


def build_dynamic_split_axis_parse_case():
    """A bare axis may be dynamic, but a *split* axis (``dim @ mesh.axis``)
    participates in canonicalisation and must resolve to a static int — a dynamic
    ``DimVar`` split extent is rejected. Direct ``parse_shard_layout_sugar``
    case (returned as a parse thunk)."""
    cta = Mesh(Topology("cta", 8), Layout((8,), (1,)), names=("cta",))
    node = ast.parse("(1, S @ cta, 32, 128)", mode="eval").body

    def do_parse():
        return parse_shard_layout_sugar(
            node, lambda n: cta if n == "cta" else None, closure={"S": _S_DYN}
        )

    return do_parse


def test_reshard_sugar_rejects_dynamic_split_axis() -> None:
    assert_parse_raises(build_dynamic_split_axis_parse_case, match="static int")


# ── static-int dims in mesh / split sugar: closure resolution + diagnostics ───
# A static-extent (mesh-shape dim or split extent) accepts an int literal or a
# closure/global int; a bool or dynamic (DimVar) value is rejected with a
# static-int diagnostic (and the sugar error must surface, not be swallowed into
# ``'Mesh' object has no attribute ...``). All cases use the string-topology
# sugar ``Mesh(topology="thread", ...)`` referencing the @func-declared topology.

_MESH_DIM_W = DimVar("W", 1, 8)
_SPLIT_EXTENT_K = DimVar("K", 32, 256)


def build_closure_mesh_dims_func(warps, lanes):
    """A mesh-shape sugar (``layout=(warps, lanes)``) whose dims come from the
    enclosing closure — must resolve like the integer literals."""

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(warps, lanes), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def build_closure_split_extent_func(k_tile):
    """A split extent (``k_tile @ (m.w, m.t)``) taken from the enclosing closure
    — must resolve like the integer literal."""

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, k_tile @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def build_literal_reshard_func():
    """All-literal reference form: ``layout=(4, 32)`` mesh dims and a
    ``128 @ (m.w, m.t)`` split extent; both closure builders above print to it."""

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def test_closure_int_mesh_dims_resolve_like_literal() -> None:
    """A closure/global int in a mesh-shape sugar prints back to the literal
    form — the parser must not reject the ``ast.Name``."""
    assert as_script(build_closure_mesh_dims_func(4, 32)) == as_script(
        build_literal_reshard_func()
    )


def test_closure_int_split_extent_resolves_like_literal() -> None:
    """A closure/global int used as a split extent prints back to the literal
    form."""
    assert as_script(build_closure_split_extent_func(128)) == as_script(
        build_literal_reshard_func()
    )


def build_dimvar_mesh_dim_func():
    """A dynamic ``DimVar`` in a mesh-shape position is a static-extent
    violation and must be rejected with a static-int diagnostic."""

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(_MESH_DIM_W, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def build_bool_mesh_dim_func():
    """A ``bool`` mesh dim is rejected even though ``bool`` subclasses ``int``."""

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(True, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def build_dimvar_split_extent_func():
    """A dynamic ``DimVar`` split extent through the full ``@func`` op-arg path
    must report the static-int diagnostic, not ``'Mesh' object has no attribute
    'w'`` (the sugar error must not be swallowed)."""

    @func(topologies=(Topology("cta", 8), Topology("thread", 128)))
    def _f(x: Tensor[(8, _SPLIT_EXTENT_K), "bf16"]) -> Tensor[(8, 1), "bf16"]:
        with Mesh(topology="thread", layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (8, _SPLIT_EXTENT_K @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (8, 1), "gmem")  # noqa: F821

    return _f


def build_bool_split_extent_single_axis_func():
    """A ``bool`` split extent in the single-axis form (``True @ m.w``) is
    rejected with a static-int diagnostic."""

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, True @ m.w), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def build_bool_split_extent_multi_axis_func():
    """A ``bool`` split extent in the multi-axis form (``True @ (m.w, m.t)``) is
    rejected with a static-int diagnostic."""

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(topology="thread", layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, True @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


@pytest.mark.parametrize(
    "build_func",
    [
        build_dimvar_mesh_dim_func,
        build_bool_mesh_dim_func,
        build_dimvar_split_extent_func,
        build_bool_split_extent_single_axis_func,
        build_bool_split_extent_multi_axis_func,
    ],
    ids=[
        "dimvar-mesh-dim",
        "bool-mesh-dim",
        "dimvar-split-extent",
        "bool-split-single-axis",
        "bool-split-multi-axis",
    ],
)
def test_static_extent_position_rejects_non_static_int(build_func) -> None:
    """A dynamic (``DimVar``) or ``bool`` value in a static-extent position is
    rejected with a clear static-int diagnostic; the sugar error must surface
    rather than be swallowed into a generic attribute error."""
    assert_build_raises(build_func, match="static int")
