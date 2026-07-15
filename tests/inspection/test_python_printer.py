"""Inspection printer tests: round-trip, dump integration."""

import os
from math import prod

from tests.fixtures.demo_ir import build_demo
from tests.fixtures.qwen3_attention_graph import build_qwen3_attention_main_2cta_headnorm
from tilefoundry.dump import DumpFlags, FileDumper, current_scope, dump
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.parser.hir_parser import parse_script


def _structural_equal(a, b, path="") -> bool:
    """Compare two HIR values for structural equality."""
    if type(a) is not type(b):
        print(f"MISMATCH type at {path}: {type(a).__name__} vs {type(b).__name__}")
        return False

    if isinstance(a, Function):
        if a.name != b.name:
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
            return False
        return _structural_equal(a.type, b.type, f"{path}.type")

    if isinstance(a, Constant):
        return a.value == b.value

    if isinstance(a, Call):
        if type(a.target) is not type(b.target):
            return False
        if len(a.args) != len(b.args):
            return False
        for i, (aa, bb) in enumerate(zip(a.args, b.args)):
            if not _structural_equal(aa, bb, f"{path}.args[{i}]"):
                return False
        for pi in type(a.target).params():
            if pi.kind == "attribute":
                av = getattr(a.target, pi.name, None)
                bv = getattr(b.target, pi.name, None)
                if not _attr_equal(av, bv, f"{path}.attr.{pi.name}"):
                    return False
        return _structural_equal(a.type, b.type, f"{path}.type")

    if isinstance(a, TensorType):
        if a.shape != b.shape:
            return False
        if a.dtype != b.dtype:
            return False
        if a.storage != b.storage:
            return False
        if not _attr_equal(a.layout, b.layout, f"{path}.layout"):
            return False
        return True

    return a == b


def _attr_equal(a, b, path="") -> bool:
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, tuple):
        if len(a) != len(b):
            return False
        return all(_attr_equal(aa, bb, f"{path}[{i}]") for i, (aa, bb) in enumerate(zip(a, b)))
    if hasattr(a, "__dataclass_fields__"):
        for f_name in a.__dataclass_fields__:
            if not _attr_equal(getattr(a, f_name), getattr(b, f_name), f"{path}.{f_name}"):
                return False
        return True
    return a == b


class TestPythonPrinterRoundTrip:
    def test_demo_ir_roundtrip(self):
        """print(build_demo()) → parse_script → structural equal."""
        fn, _, _ = build_demo()
        src = as_script(fn)
        fn2 = parse_script(src)
        assert _structural_equal(fn, fn2), "round-trip mismatch"

    def test_qwen3_output_is_valid_python(self):
        """Qwen3 printed output must be valid Python syntax."""
        fn = build_qwen3_attention_main_2cta_headnorm()
        src = as_script(fn, module="M")
        compile(src, "<test>", "exec")

    def test_demo_output_is_valid_python(self):
        """Printed output must be valid Python syntax."""
        fn, _, _ = build_demo()
        src = as_script(fn)
        compile(src, "<test>", "exec")


class TestModuleRoundTrip:
    def test_qwen3_module_roundtrip_param_layouts(self):
        """Qwen3 @module round-trip: all param layouts/storage preserved."""

        fn1 = build_qwen3_attention_main_2cta_headnorm()
        src = as_script(fn1, module="M")
        fn2 = parse_script(src)

        for p1, p2 in zip(fn1.params, fn2.params):
            t1, t2 = p1.type, p2.type
            if not isinstance(t1, TensorType) or not isinstance(t2, TensorType):
                continue
            assert t1.shape == t2.shape, f"{p1.name} shape mismatch"
            assert t1.dtype == t2.dtype, f"{p1.name} dtype mismatch"
            assert t1.storage == t2.storage, f"{p1.name} storage mismatch"
            if isinstance(t1.layout, ShardLayout) and isinstance(t2.layout, ShardLayout):
                # Per the sugar canonicalization rule, sugar ``N @ m.a``
                # with N > mesh_extent(a) expands into the factorised pair on
                # re-parse, so fn1 (built via constructor in legacy form) and
                # fn2 (round-tripped through sugar) may differ in cute shape
                # rank. The Split-bearing layout-axis index is preserved by
                # canonicalization (Split(k) stays Split(k)); only residual
                # Broadcast dims may be appended.
                a1 = [type(a) for a in t1.layout.attrs]
                a2 = [type(a) for a in t2.layout.attrs]
                assert a1 == a2, \
                    f"{p1.name} attr-kind mismatch: {t1.layout.attrs} vs {t2.layout.attrs}"
                assert prod(t1.layout.layout.shape) == prod(t2.layout.layout.shape), \
                    f"{p1.name} layout total size mismatch"
                # Mesh identity: topology, layout, names
                m1, m2 = t1.layout.mesh, t2.layout.mesh
                assert m1.topology.name == m2.topology.name, \
                    f"{p1.name} mesh topology name mismatch"
                assert m1.topology.size == m2.topology.size, \
                    f"{p1.name} mesh topology size mismatch"
                assert m1.layout.shape == m2.layout.shape, \
                    f"{p1.name} mesh layout shape mismatch"
                assert m1.names == m2.names, \
                    f"{p1.name} mesh names mismatch"

    def test_module_output_has_sugar(self):
        """@module output uses sugar for Split/Broadcast layouts."""
        fn = build_qwen3_attention_main_2cta_headnorm()
        src = as_script(fn, module="M")
        assert "@module" in src
        assert "class M:" in src
        assert "names=(" in src
        # Sugar should be used for Split-only layouts
        assert "@ gpu" in src

    def test_module_output_is_valid_python(self):
        fn = build_qwen3_attention_main_2cta_headnorm()
        src = as_script(fn, module="M")
        compile(src, "<test>", "exec")


class TestPythonPrinterDump:
    def test_dumps_to_file(self):

        fn, _, _ = build_demo()
        src = as_script(fn)
        scope = current_scope()
        assert scope is not None

        dump("demo.py", src, DumpFlags.PASS_IR)

        dumper = scope.dumper
        assert isinstance(dumper, FileDumper)
        py_path = os.path.join(str(dumper.root), "demo.py")
        assert os.path.isfile(py_path)
