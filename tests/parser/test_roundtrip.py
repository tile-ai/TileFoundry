"""Parser-owned checks for canonical grid-loop bindings."""

import pytest

from tests._source import import_dsl
from tests.fixtures.placed.gqa_decode import GqaOnline
from tests.fixtures.placed.region_boundaries import RegionBoundaries
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Constant, binding_name
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.specialize import specialize_concretely
from tilefoundry.ir.visitor import collect_exprs, expr_children


def test_gqa_correction_reads_the_old_carry_and_unique_yield() -> None:
    function = specialize_concretely(GqaOnline.entry_function(), {"ctx_len": 8})
    printed = as_script(function)

    assert printed.count("        m_new = max(m, score)") == 1
    assert " = sub(m, m_new)" in printed
    lines = printed.splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.lstrip() == "for i in range(8):"
    )
    loop_indent = lines[start][: -len(lines[start].lstrip())]
    body_indent = loop_indent + "    "
    end = next(
        index
        for index in range(start + 1, len(lines))
        if lines[index].startswith(loop_indent)
        and not lines[index].startswith(body_indent)
    )
    yielded = lines[end - 3 : end]
    assert [line.split(" = ", 1)[0] for line in yielded] == [
        f"{body_indent}l",
        f"{body_indent}o",
        f"{body_indent}m",
    ]
    assert yielded[-1] == f"{body_indent}m = m_new"
    assert len({line.split(" = ", 1)[1] for line in yielded}) == 3
    assert lines[end] == f'{loop_indent}k_n = cast(k_new, dtype="f32")'


def test_region_boundaries_round_trip() -> None:
    """Printing and reparsing preserves nested scopes and escaped values."""
    once = as_script(RegionBoundaries)
    restored = import_dsl(once, "RegionBoundaries")
    canonical = as_script(restored)
    assert as_script(import_dsl(canonical, "RegionBoundaries")) == canonical

    body = restored.entry_function().body
    producer = next(
        expr
        for expr in collect_exprs(body)
        if isinstance(expr, MeshRegion) and binding_name(expr.body) == "v2"
    )
    assert sum(producer in expr_children(expr) for expr in collect_exprs(body)) == 2


@pytest.mark.parametrize(
    ("expression", "value"),
    [("-1.0", -1.0), ("-3", -3), ("-1e999", float("-inf")), ("-x", None)],
)
def test_signed_literals_are_constants_and_negated_values_are_calls(expression, value) -> None:
    """Literal signs belong to constants; negating a runtime value remains a call."""
    function = import_dsl(
        f"""from tilefoundry import func
from tilefoundry.dsl import Tensor

@func
def signed(x: Tensor[(), "f32"]):
    return {expression}
""",
        "signed",
    )
    result = function.body
    if value is None:
        assert isinstance(result, Call)
        assert isinstance(result.target, Unary)
        assert result.target.kind is UnaryKind.NEG
        assert result.args[0] is function.params[0]
    else:
        assert isinstance(result, Constant)
        assert type(result.value) is type(value)
        assert result.value == value
