"""Round-trip tests: print → parse → structural equality."""

import pytest

from tests.fixtures.demo_ir import build_demo
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import add, mul
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Constant, Op, Tuple, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.parser.hir_parser import parse_script


def _is_op(obj) -> bool:
    """Check if obj is a tilefoundry Op instance."""
    return isinstance(obj, Op)


def _attr_equal(a, b, path="") -> bool:
    """Compare two attribute values (may be nested dataclasses or Ops without __eq__)."""
    if a is b:
        return True
    if type(a) is not type(b):
        print(f"MISMATCH attr type at {path}: {type(a).__name__} vs {type(b).__name__}")
        return False
    # Handle tuples (e.g. attrs)
    if isinstance(a, tuple):
        if len(a) != len(b):
            print(f"MISMATCH tuple len at {path}: {len(a)} vs {len(b)}")
            return False
        return all(_attr_equal(aa, bb, f"{path}[{i}]") for i, (aa, bb) in enumerate(zip(a, b)))
    # Handle Op instances — compare via params()
    if _is_op(a):
        for pi in type(a).params():
            av = getattr(a, pi.name, None)
            bv = getattr(b, pi.name, None)
            if not _attr_equal(av, bv, f"{path}.{pi.name}"):
                return False
        return True
    # Handle dataclass-like objects (check all public fields)
    if hasattr(a, "__dataclass_fields__"):
        for f_name in a.__dataclass_fields__:
            av = getattr(a, f_name, None)
            bv = getattr(b, f_name, None)
            if not _attr_equal(av, bv, f"{path}.{f_name}"):
                return False
        return True
    # Fallback to equality
    return a == b


def _structural_equal(a, b, path="") -> bool:
    """Compare two HIR expressions for structural equality."""
    if type(a) is not type(b):
        print(f"MISMATCH type at {path}: {type(a).__name__} vs {type(b).__name__}")
        return False

    # Handle Function
    if isinstance(a, Function):
        if a.name != b.name:
            print(f"MISMATCH name at {path}: {a.name} vs {b.name}")
            return False
        if len(a.params) != len(b.params):
            return False
        for i, (pa, pb) in enumerate(zip(a.params, b.params)):
            if not _structural_equal(pa, pb, f"{path}.params[{i}]"):
                return False
        if not _structural_equal(a.body, b.body, f"{path}.body"):
            return False
        if not _structural_equal(a.return_type, b.return_type, f"{path}.return_type"):
            return False
        return True

    if isinstance(a, Var):
        if a.name != b.name:
            print(f"MISMATCH Var.name at {path}: {a.name} vs {b.name}")
            return False
        return _structural_equal(a.type, b.type, f"{path}.type")

    if isinstance(a, Constant):
        return a.value == b.value

    if isinstance(a, Call):
        if type(a.target) is not type(b.target):
            print(f"MISMATCH target at {path}: {type(a.target).__name__} vs {type(b.target).__name__}")
            return False
        if len(a.args) != len(b.args):
            print(f"MISMATCH arg count at {path}: {len(a.args)} vs {len(b.args)}")
            return False
        for i, (aa, bb) in enumerate(zip(a.args, b.args)):
            if not _structural_equal(aa, bb, f"{path}.args[{i}]"):
                return False
        # Compare op attributes (Op dataclasses may not have __eq__)
        for pi in type(a.target).params():
            if pi.kind == "attribute":
                av = getattr(a.target, pi.name, None)
                bv = getattr(b.target, pi.name, None)
                if not _attr_equal(av, bv, f"{path}.attr.{pi.name}"):
                    return False
        return _structural_equal(a.type, b.type, f"{path}.type")

    if isinstance(a, Tuple):
        if len(a.elements) != len(b.elements):
            print(f"MISMATCH tuple element count at {path}: {len(a.elements)} vs {len(b.elements)}")
            return False
        for i, (ea, eb) in enumerate(zip(a.elements, b.elements)):
            if not _structural_equal(ea, eb, f"{path}.elements[{i}]"):
                return False
        return _structural_equal(a.type, b.type, f"{path}.type")

    if isinstance(a, TupleType):
        if len(a.fields) != len(b.fields):
            print(f"MISMATCH tuple field count at {path}: {len(a.fields)} vs {len(b.fields)}")
            return False
        return all(
            _structural_equal(fa, fb, f"{path}.fields[{i}]")
            for i, (fa, fb) in enumerate(zip(a.fields, b.fields))
        )

    if isinstance(a, TensorType):
        if a.shape != b.shape:
            print(f"MISMATCH shape at {path}: {a.shape} vs {b.shape}")
            return False
        if a.dtype != b.dtype:
            print(f"MISMATCH dtype at {path}: {a.dtype} vs {b.dtype}")
            return False
        if a.storage != b.storage:
            print(f"MISMATCH storage at {path}: {a.storage} vs {b.storage}")
            return False
        if not _attr_equal(a.layout, b.layout, f"{path}.layout"):
            return False
        return True

    return a == b


class TestRoundTrip:
    def test_demo_ir_roundtrip_with_keyword_attrs(self):
        """print(build_demo()) → parse_script → structural equal."""
        fn, _, _ = build_demo()
        src = as_script(fn)
        fn2 = parse_script(src)
        assert _structural_equal(fn, fn2), "round-trip mismatch with keyword attrs"

    def test_demo_ir_roundtrip_with_positional_attrs(self):
        """reshard(a, shared_layout, storage) without layout= keyword."""
        src = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "from tilefoundry.ir.types.shard import (\n"
            "    B, S, P, Layout, Mesh, Layout, ShardLayout, Topology,\n"
            ")\n"
            "shared_layout = ShardLayout(\n"
            "    layout=Layout((1, 1536), (1536, 1)),\n"
            "    attrs=(),\n"
            '    mesh=Mesh(Topology("cta", 128), Layout((128,), (1,))),\n'
            ")\n"
            "\n"
            "@func\n"
            'def test_pos(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:\n'
            '    b = reshard(a, shared_layout, "smem")\n'
            "    return b\n"
        )
        fn = parse_script(src)
        assert fn.name == "test_pos"
        # Check that the body is a reshard with correct args
        body = fn.body
        assert isinstance(body, Call)
        assert isinstance(body.target, Reshard)
        assert body.target.storage == StorageKind.SMEM
        # layout attribute should be set
        assert body.target.layout is not None

    def test_positional_and_keyword_equivalent(self):
        """reshard(a, shared_layout) ≡ reshard(a, layout=shared_layout)."""
        src_pos = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "from tilefoundry.ir.types.shard import (\n"
            "    B, S, P, Layout, Mesh, Layout, ShardLayout, Topology,\n"
            ")\n"
            "sl = ShardLayout(\n"
            "    layout=Layout((1, 1536), (1536, 1)),\n"
            "    attrs=(),\n"
            '    mesh=Mesh(Topology("cta", 128), Layout((128,), (1,))),\n'
            ")\n"
            "\n"
            "@func\n"
            'def pos(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:\n'
            "    b = reshard(a, sl)\n"
            "    return b\n"
        )
        src_kw = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "from tilefoundry.ir.types.shard import (\n"
            "    B, S, P, Layout, Mesh, Layout, ShardLayout, Topology,\n"
            ")\n"
            "sl = ShardLayout(\n"
            "    layout=Layout((1, 1536), (1536, 1)),\n"
            "    attrs=(),\n"
            '    mesh=Mesh(Topology("cta", 128), Layout((128,), (1,))),\n'
            ")\n"
            "\n"
            "@func\n"
            'def kw(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:\n'
            "    b = reshard(a, layout=sl)\n"
            "    return b\n"
        )
        fn_pos = parse_script(src_pos)
        fn_kw = parse_script(src_kw)
        # Compare bodies structurally (ignore function name)
        assert _structural_equal(fn_pos.body, fn_kw.body), \
            "positional and keyword should produce equivalent IR"

    def test_too_many_positional_args_errors(self):
        """reshard(a, sl, 'smem', 1, 'extra') — 5 positional, only 4 params."""

        src = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "from tilefoundry.ir.types.shard import *\n"
            "sl = ShardLayout(layout=Layout((1,), (1,)), attrs=(), mesh=Mesh(Topology('c',1), Layout((1,),(1,))))\n"
            "@func\n"
            'def f(a: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:\n'
            '    b = reshard(a, sl, "smem", 1, "extra")\n'
            "    return b\n"
        )
        with pytest.raises(VerifyError, match="too many positional"):
            parse_script(src)

    def test_duplicate_positional_and_keyword_errors(self):
        """reshard(a, sl, layout=sl2) — duplicate binding for layout."""

        src = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "from tilefoundry.ir.types.shard import *\n"
            "sl = ShardLayout(layout=Layout((1,), (1,)), attrs=(), mesh=Mesh(Topology('c',1), Layout((1,),(1,))))\n"
            "@func\n"
            'def f(a: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:\n'
            "    b = reshard(a, sl, layout=sl)\n"
            "    return b\n"
        )
        with pytest.raises(VerifyError, match="duplicate binding"):
            parse_script(src)

    def test_wrong_attr_type_errors(self):
        """reshard(a, 123) — int is not a ShardLayout, fails in typeinfer."""

        src = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "@func\n"
            'def f(a: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:\n'
            "    b = reshard(a, 123)\n"
            "    return b\n"
        )
        with pytest.raises(VerifyError, match="ShardLayout"):
            parse_script(src)

    def test_missing_input_errors(self):
        """reshard() — missing required input x and layout, TypeError."""
        src = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "@func\n"
            'def f(a: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:\n'
            "    b = reshard()\n"
            "    return b\n"
        )
        with pytest.raises((TypeError,)):
            parse_script(src)


@func
def _tuple_ret(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]):
    return add(a, b), mul(a, b)


def test_literal_tuple_return_roundtrips() -> None:
    """A tuple-returning @func prints a literal ``return (e0, e1)`` and
    re-parses to a structurally equal function (print → parse → equal)."""
    src = as_script(_tuple_ret)
    assert "return (" in src, f"tuple return not rendered:\n{src}"
    fn2 = parse_script(src)
    assert _structural_equal(_tuple_ret, fn2), f"round-trip mismatch:\n{src}"


def test_reduce_kind_roundtrips_as_dsl_string() -> None:
    """A ``reduce`` op's ``ReduceKind`` attribute prints as its DSL string value
    (``kind="sum"``), not ``ReduceKind.SUM`` (which the script does not import),
    so the printed source re-parses and re-prints identically."""
    src = (
        "from __future__ import annotations\n"
        "from tilefoundry import func\n"
        "from tilefoundry.dsl.tf import *\n"
        "from tilefoundry.dsl import Tensor\n"
        "\n"
        "@func\n"
        'def rd(x: Tensor[(2, 4), "f32"]):\n'
        '    res = reduce(x, axes=(1,), keepdim=False, kind="sum")\n'
        "    return res\n"
    )
    fn = parse_script(src)
    script = as_script(fn)
    assert 'kind="sum"' in script, script
    assert "ReduceKind" not in script, script
    reparsed = parse_script(script)
    assert as_script(reparsed) == script


def test_insert_slice_tuple_offset_arg_roundtrips() -> None:
    """A rank-3 ``insert_slice`` whose offset is a literal tuple argument prints
    the tuple inline as a literal ``(e0, e1, e2)`` at the call site (the parser
    lifts an inline offset tuple back to a hir Tuple), so the source re-parses
    without a dangling reference and re-prints identically."""
    src = (
        "from __future__ import annotations\n"
        "from tilefoundry import func\n"
        "from tilefoundry.dsl.tf import *\n"
        "from tilefoundry.dsl import Tensor\n"
        "\n"
        "@func\n"
        'def ins(dst: Tensor[(2, 5, 3), "f32"], upd: Tensor[(2, 1, 3), "f32"]):\n'
        "    res = insert_slice(dst, upd, (0, 1, 0))\n"
        "    return res\n"
    )
    fn = parse_script(src)
    script = as_script(fn)
    reparsed = parse_script(script)
    assert as_script(reparsed) == script


def test_shadowed_call_loc_roundtrips() -> None:
    """When a call's source loc collides with an op name (``vals, idx = topk``
    gives the ``topk`` call loc ``"topk"``), the printer renames the binding to
    ``topk_out`` to avoid shadowing the op; the emitted ``# loc`` comment must
    reflect the emitted binding, so the source re-parses and re-prints identically."""
    src = (
        "from __future__ import annotations\n"
        "from tilefoundry import func\n"
        "from tilefoundry.dsl.tf import *\n"
        "from tilefoundry.dsl import Tensor\n"
        "\n"
        "@func\n"
        'def sh(x: Tensor[(4, 8), "f32"]):\n'
        "    vals, idx = topk(x, k=3, axis=-1, largest=True, sorted=True)\n"
        "    return vals\n"
    )
    fn = parse_script(src)
    script = as_script(fn)
    assert 'topk_out = topk(' in script, script
    assert 'loc="topk_out"' in script and 'loc="topk"' not in script, script
    reparsed = parse_script(script)
    assert as_script(reparsed) == script


def test_low_precision_dtype_names_roundtrip() -> None:
    """A ``@func`` whose parameters are typed with the three low-precision dtype
    names (fp8e4m3, f8e8m0, f4e2m1) prints those names and re-parses to the same
    dtypes (print → parse → structural equal)."""
    src = (
        "from __future__ import annotations\n"
        "from tilefoundry import func\n"
        "from tilefoundry.dsl.tf import *\n"
        "from tilefoundry.dsl import Tensor\n"
        "\n"
        "@func\n"
        'def lp(a: Tensor[(4,), "fp8e4m3"], b: Tensor[(4,), "f8e8m0"], '
        'c: Tensor[(4,), "f4e2m1"]):\n'
        "    return (a, b, c)\n"
    )
    fn = parse_script(src)
    assert [p.type.dtype for p in fn.params] == [
        DType.fp8e4m3,
        DType.f8e8m0,
        DType.f4e2m1,
    ]
    printed = as_script(fn)
    for name in ("fp8e4m3", "f8e8m0", "f4e2m1"):
        assert name in printed, printed
    fn2 = parse_script(printed)
    assert _structural_equal(fn, fn2), f"round-trip mismatch:\n{printed}"


def test_tuple_return_with_mesh_element_roundtrips() -> None:
    """A tuple-return element that introduces a mesh (a ``reshard``) must be
    discovered by the printer's mesh collection via ``Tuple.elements``; the
    rendered call references the declared mesh and round-trips."""
    src = (
        "from __future__ import annotations\n"
        "from tilefoundry import func\n"
        "from tilefoundry.dsl.tf import *\n"
        "from tilefoundry.dsl import Tensor\n"
        "from tilefoundry.ir.types.shard import (\n"
        "    B, S, P, Layout, Mesh, ShardLayout, Topology,\n"
        ")\n"
        "sl = ShardLayout(\n"
        "    layout=Layout((1, 1536), (1536, 1)),\n"
        "    attrs=(),\n"
        '    mesh=Mesh(Topology("cta", 128), Layout((128,), (1,))),\n'
        ")\n"
        "\n"
        "@func\n"
        'def f(a: Tensor[(1, 1536), "f32"], c: Tensor[(1, 1536), "f32"]):\n'
        '    b = reshard(a, sl, "smem")\n'
        "    return (b, c)\n"
    )
    fn = parse_script(src)
    printed = as_script(fn)
    assert "return (" in printed, printed
    fn2 = parse_script(printed)
    assert _structural_equal(fn, fn2), f"round-trip mismatch:\n{printed}"
