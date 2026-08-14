"""``where(...)`` schedule constraints on the HIR source surface.

A ``where`` annotation states layout / mesh / storage intent on a binding, a
parameter, or a bound tuple element. The real-model corpus carries
``where(layout=...)`` annotations with closure-resolved extents and a Broadcast
value state, so their print / re-import path is witnessed there; the cases here
are the exact metadata one canonical annotation produces, and the diagnostics for
annotations that cannot mean anything.
"""

from __future__ import annotations

from tests._source import import_dsl
from tests.parser.error_cases import where_source as _source
from tilefoundry.inspection import as_script
from tilefoundry.ir.constraints import (
    LayoutConstraint,
    MeshConstraint,
    ScheduleConstraintMetadata,
    StorageConstraint,
    constraint_metadata,
)
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Partial, Split
from tilefoundry.parser import hir_parser

_MESH_PRELUDE = (
    "from tilefoundry.ir.types.shard import Layout, Mesh, Topology\n\n"
    'cta_mesh = Mesh((Topology("cta", 8),), Layout((8,), (1,)))'
)


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


def test_constraint_metadata_attaches_to_the_existing_ssa_node(monkeypatch) -> None:
    attached = []
    original = hir_parser._HirBodyVisitor._attach_metadata

    def capture(expr, metadata):
        attached.append(expr)
        original(expr, metadata)

    monkeypatch.setattr(
        hir_parser._HirBodyVisitor, "_attach_metadata", staticmethod(capture)
    )
    function = import_dsl(
        _source(
            """    y = tf.add(x, x)
    z: where(storage="gmem") = y
    out = tf.mul(z, z)
    return out"""
        )
    )

    assert len(attached) == 1
    assert function.body.args[0] is attached[0]
    assert function.body.args[1] is attached[0]
    assert constraint_metadata(attached[0]) is not None


def test_layout_extent_names_resolve_through_the_closure_or_fail() -> None:
    """A named layout extent is read out of the function's globals.

    A named layout extent is read out of the function's globals, so it may be
    an int or a ``DimVar``. The names that resolve to neither -- or to nothing --
    are rows in ``error_cases.py``.
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
