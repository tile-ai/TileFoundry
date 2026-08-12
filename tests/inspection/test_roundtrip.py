"""Cover print-import-print fixed points beyond the model corpus.

Cases include literal tuples, shadowed names, low-precision dtypes, nested mesh
declarations, and composed layouts. Reprinting must reproduce source exactly.

See [inspection §2.7](docs/spec/inspection.md#27-round-trip-contract).
"""

from tests._source import import_dsl
from tilefoundry.inspection import as_script
from tilefoundry.ir.types import DType

_HEADER = (
    "from __future__ import annotations\n"
    "from tilefoundry import func\n"
    "from tilefoundry.dsl.tf import *\n"
    "from tilefoundry.dsl import Tensor\n"
)

_SHARD_IMPORT = (
    "from tilefoundry.ir.types.shard import (\n"
    "    B, S, P, ComposedLayout, Layout, Mesh, ShardLayout, Topology,\n"
    ")\n"
)


def test_positional_and_keyword_attrs_are_the_same_program() -> None:
    """``reshard(a, shared_layout)`` ≡ ``reshard(a, layout=shared_layout)``.

    ``reshard(a, shared_layout)`` ≡ ``reshard(a, layout=shared_layout)``: an
    attribute may be passed either way at the call site, and the printer has one
    canonical form for both, so the two sources print identically.
    """
    body = (
        "sl = ShardLayout(\n"
        "    layout=Layout((1, 1536), (1536, 1)),\n"
        "    attrs=(),\n"
        '    mesh=Mesh((Topology("cta", 128),), Layout((128,), (1,))),\n'
        ")\n"
        "\n"
        "@func\n"
        'def f(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:\n'
    )
    printed = [
        as_script(import_dsl(_HEADER + _SHARD_IMPORT + body + call))
        for call in (
            "    b = reshard(a, sl)\n    return b\n",
            "    b = reshard(a, layout=sl)\n    return b\n",
        )
    ]

    assert printed[0] == printed[1]


def test_insert_slice_tuple_offset_arg_roundtrips() -> None:
    """Test insert slice tuple offset arg roundtrips.

    A rank-3 ``insert_slice`` whose offset is a literal tuple argument prints
    the tuple inline as a literal ``(e0, e1, e2)`` at the call site (the parser
    lifts an inline offset tuple back to a hir Tuple), so importing the source
    leaves no dangling reference and re-printing is identical.
    """
    fn = import_dsl(
        _HEADER + "\n@func\n"
        'def ins(dst: Tensor[(2, 5, 3), "f32"], upd: Tensor[(2, 1, 3), "f32"]):\n'
        "    res = insert_slice(dst, upd, (0, 1, 0))\n"
        "    return res\n"
    )
    script = as_script(fn)
    assert as_script(import_dsl(script)) == script


def test_slice_runtime_starts_tuple_roundtrips() -> None:
    """An explicit Slice preserves a runtime start that subscript sugar cannot spell."""
    fn = import_dsl(
        _HEADER + "\n@func\n"
        'def cut(x: Tensor[(8, 4), "f32"], start: Tensor[(), "i64"]):\n'
        "    return slice(x, (start, 0), sizes=(2, 4), strides=(1, 1))\n"
    )
    script = as_script(fn)

    assert "slice(x, (start, 0), sizes=(2, 4), strides=(1, 1))" in script
    assert as_script(import_dsl(script)) == script


def test_two_argument_tile_window_roundtrips_as_a_subscript() -> None:
    fn = import_dsl(
        _HEADER + "\n@func\n"
        'def scan_copy(x: Tensor[(4, 4), "f32"]):\n'
        '    out = zeros(shape=(4, 4), dtype="f32")\n'
        "    for row in tile(4, 2):\n"
        "        out = insert_slice(out, x[row, :], (row, 0))\n"
        "    return out\n"
    )
    script = as_script(fn)

    assert "x[row, :]" in script
    assert as_script(import_dsl(script)) == script


def test_shadowed_call_loc_roundtrips() -> None:
    """Test shadowed call loc roundtrips.

    When a call's source loc collides with an op name (``vals, idx = topk``
    gives the ``topk`` call loc ``"topk"``), the printer renames the binding to
    ``topk_out`` to avoid shadowing the op. The renamed binding is carried by the
    left-hand side and nothing else, so re-printing the imported source is what
    proves the label survived: had it come back as ``topk``, the fixed point would
    not hold.
    """
    fn = import_dsl(
        _HEADER + "\n@func\n"
        'def sh(x: Tensor[(4, 8), "f32"]):\n'
        "    vals, idx = topk(x, k=3, axis=-1, largest=True, sorted=True)\n"
        "    return vals\n"
    )
    script = as_script(fn)
    assert "topk_out = topk(" in script, script
    assert as_script(import_dsl(script)) == script


def test_low_precision_dtype_names_roundtrip() -> None:
    """Test low precision dtype names roundtrip.

    A ``@func`` whose parameters are typed with the three low-precision dtype
    names (fp8e4m3, f8e8m0, f4e2m1) prints those names and imports with the same
    dtypes.
    """
    expected = [DType.fp8e4m3, DType.f8e8m0, DType.f4e2m1]
    fn = import_dsl(
        _HEADER + "\n@func\n"
        'def lp(a: Tensor[(4,), "fp8e4m3"], b: Tensor[(4,), "f8e8m0"], '
        'c: Tensor[(4,), "f4e2m1"]):\n'
        "    return (a, b, c)\n"
    )
    assert [p.type.dtype for p in fn.params] == expected

    printed = as_script(fn)
    for name in ("fp8e4m3", "f8e8m0", "f4e2m1"):
        assert name in printed, printed
    imported = import_dsl(printed)
    assert [p.type.dtype for p in imported.params] == expected
    assert as_script(imported) == printed


def test_tuple_return_with_mesh_element_roundtrips() -> None:
    """Test tuple return with mesh element roundtrips.

    A tuple-return element that introduces a mesh (a ``reshard``) must be
    discovered by the printer's mesh collection via ``Tuple.elements``; the
    rendered call references the declared mesh and round-trips.
    """
    fn = import_dsl(
        _HEADER + _SHARD_IMPORT + "sl = ShardLayout(\n"
        "    layout=Layout((1, 1536), (1536, 1)),\n"
        "    attrs=(),\n"
        '    mesh=Mesh((Topology("cta", 128),), Layout((128,), (1,))),\n'
        ")\n"
        "\n@func\n"
        'def f(a: Tensor[(1, 1536), "f32"], c: Tensor[(1, 1536), "f32"]):\n'
        '    b = reshard(a, sl, "smem")\n'
        "    return (b, c)\n"
    )
    printed = as_script(fn)
    assert "return (" in printed, printed
    assert "mesh=Mesh(" in printed, printed
    assert as_script(import_dsl(printed)) == printed


def test_nested_composed_shard_layout_roundtrips_without_flattening() -> None:
    """A ``ComposedLayout`` whose outer is a prior ``ShardLayout`` stage must print as that nesting.

    A ``ComposedLayout`` whose outer is a prior ``ShardLayout`` stage must print
    as that nesting: flattening it would lose which mesh level owns which axis.
    """
    fn = import_dsl(
        "from __future__ import annotations\n"
        "from tilefoundry import func\n"
        "from tilefoundry.dsl import Tensor\n"
        + _SHARD_IMPORT
        + "thread = Mesh((Topology('thread', 2),), Layout((2,), (1,)))\n"
        "cta = Mesh((Topology('cta', 4),), Layout((4,), (1,)))\n"
        "prior = ShardLayout(\n"
        "    layout=Layout((8,), (1,)), attrs=(S(0),), mesh=thread,\n"
        ")\n"
        "nested = ShardLayout(\n"
        "    layout=ComposedLayout(inner=None, offset=1, outer=prior),\n"
        "    attrs=(B(),),\n"
        "    mesh=cta,\n"
        ")\n"
        "\n@func\n"
        "def f(a: Tensor[(8,), 'f32', nested]) -> Tensor[(8,), 'f32', nested]:\n"
        "    return a\n"
    )
    printed = as_script(fn)

    assert "layout=ComposedLayout(" in printed
    assert "outer=ShardLayout(" in printed
    assert as_script(import_dsl(printed)) == printed


def test_a_loop_used_as_a_value_prints_the_name_its_carry_has() -> None:
    """A ``for`` statement binds no name of its own.

    A ``for`` statement binds no name of its own. A loop with one carried
    value, consumed by a later statement, has to render as that carry — a
    dangling reference does not survive importing the emitted file.
    """
    fn = import_dsl(
        _HEADER + "\n@func\n"
        'def acc(x: Tensor[(4, 8), "f32"]):\n'
        '    total = zeros(shape=(4, 8), dtype="f32")\n'
        "    for i in tile(4):\n"
        "        total = add(total, x)\n"
        "    scaled = mul(total, x)\n"
        "    return scaled\n"
    )
    printed = as_script(fn)

    assert "mul(total, x)" in printed, printed
    assert as_script(import_dsl(printed)) == printed
