"""Shard-layout sugar parse tests.

Each scenario is a named ``build_*_func`` / ``build_*_case`` builder carrying a
docstring that describes the DSL scene; the ``test_*`` below runs it and asserts
the parsed ``ShardLayout`` or the diagnostic it must raise. Covers inline
``Split``, the ``{...}`` ``Partial`` value-state set, default ``Broadcast``,
multi-mesh-axis split, explicit strides, the single-axis ``int @ mesh``
shorthand, closure/dynamic axis extents, and the printer's fallback when a mesh
cannot be named.
"""

from __future__ import annotations

import ast

import pytest

from tests.fixtures.demo_ir import build_demo
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- binds bare op names (reshard, ...)
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import (
    Layout,
    Mesh,
    P,
    ShardLayout,
    Topology,
)
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, Split
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.parser.sugar import parse_shard_layout_sugar

_M_GPU = Mesh(
    (Topology("gpu", 8192),),
    Layout((32, 2, 8, 32), (2048, 1024, 32, 1)),
    names=("cluster", "cta", "warp", "lane"),
)
_M_MULTI = Mesh((Topology("thread", 6 * 32),), Layout((6, 32), (32, 1)), names=("w", "t"))
_M_STRIDED = Mesh((Topology("thread", 4 * 32),), Layout((4, 32), (32, 1)), names=("y", "t"))
_M_CTA = Mesh((Topology("cta", 128),), Layout((128,), (1,)), names=("cta",))
_M_STATE = Mesh(
    (Topology("thread", 4 * 2 * 16),), Layout((4, 2, 16), (32, 16, 1)), names=("l", "g", "t")
)


def build_split_inline_and_default_broadcast_func():
    """``dim @ mesh.axis`` binds a Split on that layout axis.

    ``dim @ mesh.axis`` binds a Split on that layout axis; mesh axes named in
    no Split default to Broadcast; layout strides auto-fill C-order.
    """

    @func
    def _f(
        a: Tensor[(32, 128), bf16, (32 @ _M_GPU.cluster, 2 @ _M_GPU.cta, 64), "smem"],
    ) -> Tensor[(32, 128), "f32"]:
        return a

    return _f


def build_partial_brace_value_state_func():
    """Build partial brace value state func.

    The optional final ``{mesh.axis @ P("reduction")}`` set carries a
    mesh-axis Partial value state; the layout tuple holds only Split placement,
    and unnamed axes stay Broadcast.
    """

    @func
    def _f(
        a: Tensor[(64, 128), bf16, ((32 @ _M_GPU.cluster, 64), {_M_GPU.warp @ P("sum")}), "smem"],
    ) -> Tensor[(64, 128), "f32"]:
        return a

    return _f


def build_multi_axis_split_with_remainder_func():
    """Build multi axis split with remainder func.

    ``1536 @ (w, t)`` factorises the dim into the mesh extents (6, 32) plus a
    remainder (8), each extent bound as a Split; the leading unit axis is kept.
    """

    @func
    def _f(
        a: Tensor[(1, 1536), "f32", (1, 1536 @ (_M_MULTI.w, _M_MULTI.t)), "smem"],
    ) -> Tensor[(1, 1536), "f32"]:
        return a

    return _f


def build_explicit_strides_func():
    """The ``((dims), (strides))`` form preserves user-supplied dims and strides.

    The ``((dims), (strides))`` form preserves user-supplied dims and strides;
    the explicit-strides path does not trigger single-axis canonicalization.
    """

    @func
    def _f(
        a: Tensor[(12, 4), "f32", ((12 @ _M_STRIDED.y, 4), (4, 1)), "smem"],
    ) -> Tensor[(12, 4), "f32"]:
        return a

    return _f


def build_int_at_single_axis_mesh_func():
    """Build int at single axis mesh func.

    On a single-axis mesh, ``8192 @ cta`` (extent 128) canonicalises into
    ``(128, 64)`` with the mesh axis bound as a Split on the new layout axis.
    """

    @func
    def _f(
        a: Tensor[(1, 8192), "f32", (1, 8192 @ _M_CTA), "smem"],
    ) -> Tensor[(1, 8192), "f32"]:
        return a

    return _f


@pytest.mark.parametrize(
    ("build_func", "shape", "layout", "attrs", "mesh"),
    [
        (
            build_split_inline_and_default_broadcast_func,
            (32, 128),
            Layout((32, 2, 64), (128, 64, 1)),
            (Split(0), Split(1), Broadcast(), Broadcast()),
            _M_GPU,
        ),
        (
            build_partial_brace_value_state_func,
            (64, 128),
            Layout((32, 64), (64, 1)),
            (Split(0), Broadcast(), Partial("sum"), Broadcast()),
            _M_GPU,
        ),
        (
            build_multi_axis_split_with_remainder_func,
            (1, 1536),
            Layout((1, 6, 32, 8), (1536, 256, 8, 1)),
            (Split(1), Split(2)),
            _M_MULTI,
        ),
        (
            build_explicit_strides_func,
            (12, 4),
            Layout((12, 4), (4, 1)),
            (Split(0), Broadcast()),
            _M_STRIDED,
        ),
        (
            build_int_at_single_axis_mesh_func,
            (1, 8192),
            Layout((1, 128, 64), (8192, 64, 1)),
            (Split(1),),
            _M_CTA,
        ),
    ],
    ids=[
        "split-inline-default-broadcast",
        "partial-brace-value-state",
        "multi-axis-remainder",
        "explicit-strides",
        "int-at-single-axis-mesh",
    ],
)
def test_annotation_sugar_parses_to_the_hand_written_layout(
    build_func, shape, layout, attrs, mesh
) -> None:
    """Each sugar form must land on exactly the type its verbose spelling would produce.

    Each sugar form must land on exactly the type its verbose spelling would
    produce: the logical shape stays un-factorised while the layout carries the
    factorisation, and every mesh axis gets an attr. The scene each case covers is
    in its builder's docstring.
    """
    parsed = build_func().params[0].type
    assert parsed.shape == shape
    assert parsed.storage is StorageKind.SMEM
    assert parsed.layout == ShardLayout(layout=layout, attrs=attrs, mesh=mesh)


def build_multi_axis_split_not_divisible_func():
    """A dim must be divisible by the product of the mesh extents; ``100 @`` is rejected.

    A dim must be divisible by the product of the mesh extents; ``100 @
    (w, t)`` (product 192) is rejected.
    """

    @func
    def _bad(
        a: Tensor[(1, 100), "f32", (1, 100 @ (_M_MULTI.w, _M_MULTI.t)), "smem"],
    ) -> Tensor[(1, 100), "f32"]:
        return a

    return _bad


def build_value_state_not_final_func():
    """The ``{...}`` value-state set is valid only as the last outer item.

    The ``{...}`` value-state set is valid only as the last outer item; a
    stride tuple after it is rejected.
    """

    @func
    def _bad(
        a: Tensor[
            (4, 64),
            "f32",
            ((4 @ _M_STATE.l, 64), {_M_STATE.t @ P("sum")}, (64, 1)),
            "smem",
        ],
    ) -> Tensor[(4, 64), "f32"]:
        return a

    return _bad


def build_value_state_bare_p_func():
    """``P(...)`` in the value-state set requires its reduction argument.

    ``P(...)`` in the value-state set requires its reduction argument; bare
    ``P()`` is rejected (the surface is ``mesh.axis @ P("reduction")``).
    """

    @func
    def _bad(
        a: Tensor[(4, 64), "f32", ((4 @ _M_STATE.l, 64), {_M_STATE.t @ P()}), "smem"],
    ) -> Tensor[(4, 64), "f32"]:
        return a

    return _bad


@pytest.mark.parametrize(
    ("build_func", "match"),
    [
        (build_multi_axis_split_not_divisible_func, "not divisible"),
        (build_value_state_not_final_func, "last outer item"),
        (build_value_state_bare_p_func, "reduction argument"),
    ],
    ids=["not-divisible", "value-state-not-final", "bare-p"],
)
def test_invalid_annotation_sugar_raises(build_func, match) -> None:

    with pytest.raises(ValueError, match=match):
        build_func()


def test_printer_falls_back_to_verbose_when_mesh_has_no_names() -> None:
    """A mesh without ``names=`` cannot use ``@`` sugar.

    A mesh without ``names=`` cannot use ``@`` sugar; the printer must emit the
    verbose ``ShardLayout(...)`` form instead.
    """
    fn, _, _ = build_demo()
    src = as_script(fn)
    assert "@" not in src.split("@func")[1].split("def ")[0]
    assert "ShardLayout(" in src


_S_DYN = DimVar("seq_len", 1, 4)
_MESH_DIM_W = DimVar("W", 1, 8)


def build_dynamic_bare_and_closure_split_func():
    """Build dynamic bare and closure split func.

    A reshard layout sugar may carry a dynamic ``DimVar`` bare axis (``S``)
    and a closure-resolved Name split extent (``_HQ``). The split axis is
    canonicalised against the mesh extent; the dynamic axis rides through as a
    Broadcast dim. The reshard's logical result keeps the un-factorised shape and
    strides defer to typeinfer.
    """
    _HQ, _D = 32, 128

    @func(topologies=(Topology("cta", 8),))
    def _f(
        q: Tensor[(1, _S_DYN, _HQ, _D), "bf16"],
    ) -> Tensor[(1, _S_DYN, _HQ, _D), "bf16"]:
        with Mesh(("cta",), layout=Layout((8,), (1,))) as cta:
            return reshard(q, layout=(1, _S_DYN, _HQ @ cta, _D))  # noqa: F821

    return _f


def test_reshard_sugar_accepts_dynamic_bare_and_closure_name_axis() -> None:
    body = build_dynamic_bare_and_closure_split_func().entry_function().body
    assert isinstance(body, Call) and isinstance(body.target, Reshard)

    assert body.type.shape == (1, _S_DYN, 32, 128)

    assert any(isinstance(a, Split) for a in body.target.layout.attrs)

    assert body.target.layout.layout.strides is None


def test_reshard_sugar_rejects_dynamic_split_axis() -> None:
    """Test reshard sugar rejects dynamic split axis.

    A bare axis may be dynamic, but a *split* axis (``dim @ mesh.axis``)
    participates in canonicalisation and must resolve to a static int — a dynamic
    ``DimVar`` split extent is rejected. Driven through
    ``parse_shard_layout_sugar`` directly, the layer that canonicalises.
    """
    cta = Mesh((Topology("cta", 8),), Layout((8,), (1,)), names=("cta",))
    node = ast.parse("(1, S @ cta, 32, 128)", mode="eval").body
    with pytest.raises(ValueError, match="static int"):
        parse_shard_layout_sugar(node, lambda n: cta if n == "cta" else None, closure={"S": _S_DYN})


def build_mesh_dims_reshard_func(warps, lanes):
    """Build mesh dims reshard func.

    A mesh-shape sugar (``layout=(warps, lanes)``) whose dims may be integer
    literals or closure Names — a closure int must resolve like the literal, and a
    dynamic ``DimVar`` in that static-extent position must be rejected.
    """

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(("thread",), layout=(warps, lanes), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def build_literal_reshard_func():
    """All-literal reference form.

    All-literal reference form: ``layout=(4, 32)`` mesh dims and a
    ``128 @ (m.w, m.t)`` split extent; the closure builder above prints to it.
    """

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(("thread",), layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, 128 @ (m.w, m.t)), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


def test_closure_int_mesh_dims_resolve_like_literal() -> None:
    """A closure/global int in a mesh-shape sugar prints back to the literal form.

    A closure/global int in a mesh-shape sugar prints back to the literal
    form — the parser must resolve the ``ast.Name`` rather than reject it, which
    is the positive counterpart of the static-extent diagnostic below.
    """
    assert as_script(build_mesh_dims_reshard_func(4, 32)) == as_script(build_literal_reshard_func())


def build_bool_split_extent_single_axis_func():
    """A ``bool`` split extent in the single-axis form is rejected with a static-int diagnostic.

    A ``bool`` split extent in the single-axis form (``True @ m.w``) is
    rejected with a static-int diagnostic.
    """

    @func(topologies=(Topology("thread", 128),))
    def _f(x: Tensor[(1, 128), "bf16"]) -> Tensor[(1, 128), "bf16"]:
        with Mesh(("thread",), layout=(4, 32), names=("w", "t")) as m:
            xr = reshard(x, (1, True @ m.w), "rmem")  # noqa: F821
            return reshard(xr, (1, 128), "gmem")  # noqa: F821

    return _f


@pytest.mark.parametrize(
    "build_func",
    [
        lambda: build_mesh_dims_reshard_func(_MESH_DIM_W, 32),
        build_bool_split_extent_single_axis_func,
    ],
    ids=["dimvar-mesh-dim", "bool-split-single-axis"],
)
def test_static_extent_position_rejects_non_static_int(build_func) -> None:
    """Test static extent position rejects non static int.

    A dynamic (``DimVar``) or ``bool`` value in a static-extent position is
    rejected with a clear static-int diagnostic; the sugar error must surface
    rather than be swallowed into a generic attribute error.
    """
    with pytest.raises(ValueError, match="static int"):
        build_func()
