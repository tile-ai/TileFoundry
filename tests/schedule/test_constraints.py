from __future__ import annotations

from dataclasses import replace

import pytest

from tilefoundry import VerifyError
from tilefoundry.compile import lower
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.module import Module
from tilefoundry.parser import parse_func_source
from tilefoundry.schedule.constraints import (
    AgentConstraintsMetadata,
    LayoutConstraint,
    LayoutDimKind,
    PartialConstraint,
)


def _constraint_signature(metadata):
    return tuple(
        (type(constraint), tuple(getattr(constraint, "dims", ())),
         getattr(constraint, "reduction", None), getattr(constraint, "topology", None))
        for constraint in metadata.constraints
    )


def _source(body: str) -> str:
    return f'''from __future__ import annotations
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf

@func
def candidate(x: Tensor[(8, 16), "bf16"]) -> Tensor[(8, 16), "bf16"]:
{body}
'''


def test_inline_and_standalone_constraints_attach_to_shared_ssa():
    fn = parse_func_source(
        _source(
            """    y: where(layout=(_, H @ cta)) = tf.add(x, x)
    y: where(partial=P("sum"))
    z = y
    return z"""
        )
    )

    assert isinstance(fn.body, Call)
    assert isinstance(fn.body.metadata[0], AgentConstraintsMetadata)
    metadata = fn.body.metadata[0]
    assert len(metadata.constraints) == 2
    assert isinstance(metadata.constraints[0], LayoutConstraint)
    assert metadata.constraints[0].dims[0].kind is LayoutDimKind.UNCONSTRAINED
    assert metadata.constraints[0].dims[1].kind is LayoutDimKind.SPLIT
    assert isinstance(metadata.constraints[1], PartialConstraint)

    printed = as_script(fn)
    assert "y: where(layout=(_, H @ cta))" in printed
    assert 'y: where(partial=P("sum"))' in printed
    reparsed = parse_func_source(printed)
    assert _constraint_signature(reparsed.body.metadata[0]) == _constraint_signature(
        fn.body.metadata[0]
    )


def test_parameter_constraint_prints_at_function_body_start():
    fn = parse_func_source(
        _source(
            """    x: where(layout=(D, _))
    return x"""
        )
    )
    param_metadata = fn.params[0].metadata[0]
    assert isinstance(param_metadata, AgentConstraintsMetadata)
    printed = as_script(fn)
    assert printed.index("x: where(layout=(D, _))") < printed.index("return x")
    reparsed = parse_func_source(printed)
    assert _constraint_signature(reparsed.params[0].metadata[0]) == _constraint_signature(
        fn.params[0].metadata[0]
    )


def test_metadata_is_ignored_by_expr_equality_and_hashing():
    fn = parse_func_source(_source("    return tf.add(x, x)"))
    expr = fn.body
    tagged = replace(
        expr,
        metadata=(
            AgentConstraintsMetadata(
                constraints=(LayoutConstraint(dims=()),),
            ),
        ),
    )
    assert tagged == expr
    assert hash(tagged) == hash(expr)


@pytest.mark.parametrize(
    "annotation",
    [
        "require(layout=(_, H @ cta))",
        "where(storage=\"rmem\")",
        "where(foo=(_ , H @ cta))",
        "where(layout=())",
    ],
)
def test_removed_or_invalid_constraint_forms_fail(annotation):
    with pytest.raises(VerifyError):
        parse_func_source(_source(f"    y: {annotation} = tf.add(x, x)\n    return y"))


def test_lower_rejects_unresolved_agent_constraints() -> None:
    fn = parse_func_source(
        _source(
            """    y: where(layout=(_, H @ cta)) = tf.add(x, x)
    return y"""
        )
    )
    with pytest.raises(ValueError, match="unresolved Agent Constraints"):
        lower(Module(name="candidate", functions=(fn,), entry="candidate"))
