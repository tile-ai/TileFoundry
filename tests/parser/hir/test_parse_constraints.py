"""``where(...)`` schedule constraints on the HIR source surface.

A ``where`` annotation states layout / mesh / storage intent on a binding, a
parameter, or a bound tuple element. The real-model corpus carries
``where(layout=...)`` annotations with closure-resolved extents and a Broadcast
value state, so their print / re-import path is witnessed there; the cases here
are the exact metadata one canonical annotation produces, and the diagnostics for
annotations that cannot mean anything.
"""

from __future__ import annotations

import pytest

from tests._source import import_dsl
from tilefoundry.inspection import as_script
from tilefoundry.ir.constraints import (
    LayoutConstraint,
    MeshConstraint,
    ScheduleConstraintMetadata,
    StorageConstraint,
    constraint_metadata,
)
from tilefoundry.ir.core import Call, VerifyError
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Partial, Split

_MESH_PRELUDE = (
    "from tilefoundry.ir.types.shard import Layout, Mesh, Topology\n\n"
    'cta_mesh = Mesh((Topology("cta", 8),), Layout((8,), (1,)))'
)


def _source(body: str, preamble: str = _MESH_PRELUDE, ret: str = '(8, 16), "bf16"') -> str:
    """A one-``@func`` script.

    A one-``@func`` script; *preamble* holds module-level (closure) bindings,
    for names the parser resolves via ``fn.__globals__`` rather than the SSA env.
    """
    return f"""from __future__ import annotations
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf

{preamble}

@func
def candidate(x: Tensor[(8, 16), "bf16"]) -> Tensor[{ret}]:
{body}
"""


def _layout_of(fn) -> LayoutConstraint:
    layout = constraint_metadata(fn.body).constraints[0]
    assert isinstance(layout, LayoutConstraint)
    return layout


def test_layout_mesh_storage_constraints_parse_verify_and_round_trip() -> None:
    fn = import_dsl(
        _source(
            """    y: where(layout=(_, 16 @ cta), mesh=cta_mesh, storage="gmem") = tf.add(x, x)
    return y"""
        )
    )
    verify_function(fn)

    metadata = constraint_metadata(fn.body)
    assert isinstance(metadata, ScheduleConstraintMetadata)
    assert len(metadata.constraints) == 3
    layout, mesh, storage = metadata.constraints
    assert isinstance(layout, LayoutConstraint)
    assert repr(layout.layout.shape[0]) == "_"
    assert layout.layout.shape[1] == 16
    assert layout.bindings == (("cta", Split(1)),)
    assert isinstance(mesh, MeshConstraint)
    assert isinstance(storage, StorageConstraint)

    printed = as_script(fn)
    assert "where(layout=(_, 16 @ cta), mesh=Mesh(" in printed
    assert 'storage="gmem"' in printed
    imported = import_dsl(printed)
    verify_function(imported)
    again = constraint_metadata(imported.body).constraints
    assert [type(c) for c in again] == [LayoutConstraint, MeshConstraint, StorageConstraint]
    assert again[0].bindings == layout.bindings
    assert again[0].layout.shape[1] == layout.layout.shape[1]
    assert again[2].storage == storage.storage


def test_layout_extent_names_resolve_through_the_closure_or_fail() -> None:
    """A named layout extent is read out of the function's globals.

    A named layout extent is read out of the function's globals, so it may be
    an int or a ``DimVar``; a name that resolves to neither -- or to nothing --
    must fail at the annotation rather than silently drop the extent.
    """
    body = """    y: where(layout=(_, N @ cta)) = tf.add(x, x)
    return y"""
    as_int = _layout_of(import_dsl(_source(body, preamble=_MESH_PRELUDE + "\nN = 16")))
    assert repr(as_int.layout.shape[0]) == "_"
    assert as_int.layout.shape[1] == 16

    dim_var_preamble = (
        _MESH_PRELUDE + '\nfrom tilefoundry.ir.types.dim import DimVar\nN = DimVar("S", 1, 128)'
    )
    as_dim_var = _layout_of(import_dsl(_source(body, preamble=dim_var_preamble)))
    assert isinstance(as_dim_var.layout.shape[1], DimVar)
    assert as_dim_var.layout.shape[1].name == "S"

    with pytest.raises(VerifyError, match="undefined name|where layout extent"):
        import_dsl(_source(body))
    with pytest.raises(VerifyError, match="must resolve to an int or DimVar"):
        import_dsl(_source(body, preamble=_MESH_PRELUDE + '\nN = "not-an-int"'))


def test_partial_value_state_and_the_subjects_that_accept_a_constraint() -> None:
    """Test partial value state and the subjects that accept a constraint.

    The ``{mesh @ P(...)}`` set reuses the existing shard attrs rather than a
    parser-local value state, and prints back as it was written. Besides a bound
    name, a parameter and a bound tuple element are constraint subjects too -- the
    tuple element is what lets a multi-output op's second result carry intent.
    """
    partial = import_dsl(
        _source(
            """    y: where(layout=((_, 16), {cta @ P("sum")})) = tf.add(x, x)
    return y"""
        )
    )
    assert _layout_of(partial).bindings == (("cta", Partial("sum")),)
    assert 'layout=((_, 16), {cta @ P("sum")})' in as_script(partial)

    parameter = import_dsl(
        _source("""    x: where(storage="smem")
    return x""")
    )
    assert isinstance(constraint_metadata(parameter.params[0]), ScheduleConstraintMetadata)

    tuple_fn = import_dsl(
        _source(
            """    values = tf.topk(x, k=4, axis=-1)
    ids = values[1]
    ids: where(storage="gmem")
    return ids""",
            preamble="",
            ret='(8, 4), "i64"',
        )
    )
    verify_function(tuple_fn)
    assert isinstance(tuple_fn.body, Call)
    assert isinstance(tuple_fn.body.type, TensorType)
    assert tuple_fn.body.target.__class__.__name__ == "TupleGetItem"
    assert constraint_metadata(tuple_fn.body) is not None


@pytest.mark.parametrize(
    "body",
    [
        "    y: where() = tf.add(x, x)\n    return y",
        "    y: where(layout=()) = tf.add(x, x)\n    return y",
        '    y: where(layout=(_, {cta @ B(), cta @ P("sum")})) = tf.add(x, x)\n    return y',
        '    y: where(partial=P("sum")) = tf.add(x, x)\n    return y',
        "    y: where(layout=(1.5,)) = tf.add(x, x)\n    return y",
        '    y: where(storage="gmem") = tf.add(x, x)\n    y: where(storage="gmem")\n    return y',
    ],
    ids=[
        "no-kwargs",
        "empty-layout",
        "two-bindings-one-axis",
        "unknown-kwarg",
        "non-int-extent",
        "annotated-twice",
    ],
)
def test_invalid_constraints_fail_at_source_annotation(body: str) -> None:
    with pytest.raises(VerifyError, match="where|layout|duplicate|binding"):
        import_dsl(_source(body))


def test_tuple_and_subscript_annotation_subjects_are_rejected() -> None:
    """A constraint names one value, so its subject must be a bound plain Name of tensor type.

    A constraint names one value, so its subject must be a bound plain Name of
    tensor type: a subscript lvalue and a whole tuple binding are both refused.
    """
    with pytest.raises(VerifyError, match="bound plain Name|annotation lvalue"):
        import_dsl(
            _source(
                """    value = tf.add(x, x)
    value[0]: where(storage="gmem")
    return value"""
            )
        )

    with pytest.raises(VerifyError, match="tensor-valued"):
        import_dsl(
            _source(
                """    pair = tf.topk(x, k=4, axis=-1)
    pair: where(storage="gmem")
    return x""",
                preamble="",
            )
        )
