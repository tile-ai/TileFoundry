"""Identity mutation across CUDA statement subclasses."""

from __future__ import annotations

from tests.core.ir.test_visitor import (
    _call,
    _const,
    _eval_call,
    _i32,
    _OpA,
    _var,
)
from tilefoundry.ir.tir.memory import Copy, Fill
from tilefoundry.ir.tir.stmts import (
    For,
    If,
    LetStmt,
    MeshScope,
    Return,
    Sequential,
    While,
)
from tilefoundry.ir.types import Layout, Mesh, Topology
from tilefoundry.ir.visitor import (
    StmtMutator,
)
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma
from tilefoundry_nvidia.cuda.ir.tir.nn.sm80_mma import Mma


def test_stmt_mutator_covers_all_subclasses_with_identity_invariant() -> None:
    """Every Stmt subclass round-trips through identity ``StmtMutator``."""
    i = _var("i", _i32())
    binding = _var("mvar")

    def _seq(*items) -> Sequential:
        return Sequential(body=tuple(items))

    stmts = (
        LetStmt(var=_var("y"), value=_call(_OpA(), _var("z"))),
        Return(),
        For(
            induction_var=i,
            start=_const(0, _i32()),
            stop=_const(8, _i32()),
            step=_const(1, _i32()),
            body=_seq(_eval_call(Copy(), _var("s"), _var("d"))),
        ),
        While(cond=_var("c"), body=_seq(_eval_call(Copy(), _var("s2"), _var("d2")))),
        If(cond=_var("c2"), then_body=_seq(), else_body=_seq()),
        MeshScope(
            mesh=Mesh((Topology(name="chip", size=2),), Layout((2,), (1,)), ("g",)),
            binding=binding,
            body=_seq(_eval_call(Copy(), _var("s3"), _var("d3"))),
        ),
        _eval_call(Copy(), _var("s4"), _var("d4")),
        _eval_call(Fill(), _var("t"), _const(0.0)),
        _eval_call(TiledMma(atom=Mma()), _var("L"), _var("R"), _var("A")),
        Sequential(body=()),
    )
    m = StmtMutator()
    for s in stmts:
        assert m.visit(s) is s, f"identity broken on {type(s).__name__}"
