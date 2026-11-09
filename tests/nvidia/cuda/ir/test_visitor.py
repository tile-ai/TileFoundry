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
from tilefoundry.ir import core, types
from tilefoundry.ir.tir import memory
from tilefoundry_nvidia.cuda.ir.tir import nn


def test_stmt_mutator_covers_all_subclasses_with_identity_invariant() -> None:
    """Every Stmt subclass round-trips through identity ``StmtMutator``."""
    i = _var("i", _i32())
    binding = _var("mvar")

    def _seq(*items) -> core.Sequential:
        return core.Sequential(body=tuple(items))

    stmts = (
        core.LetStmt(var=_var("y"), value=_call(_OpA(), _var("z"))),
        core.Return(),
        core.For(
            induction_var=i,
            start=_const(0, _i32()),
            stop=_const(8, _i32()),
            step=_const(1, _i32()),
            body=_seq(_eval_call(memory.Copy(), _var("s"), _var("d"))),
        ),
        core.While(cond=_var("c"), body=_seq(_eval_call(memory.Copy(), _var("s2"), _var("d2")))),
        core.If(cond=_var("c2"), then_body=_seq(), else_body=_seq()),
        core.MeshScope(
            mesh=types.Mesh((types.Topology(name="chip", size=2),), types.Layout((2,), (1,)), ("g",)),
            binding=binding,
            body=_seq(_eval_call(memory.Copy(), _var("s3"), _var("d3"))),
        ),
        _eval_call(memory.Copy(), _var("s4"), _var("d4")),
        _eval_call(memory.Fill(), _var("t"), _const(0.0)),
        _eval_call(nn.TiledMma(atom=nn.Mma()), _var("L"), _var("R"), _var("A")),
        core.Sequential(body=()),
    )
    m = core.StmtMutator()
    for s in stmts:
        assert m.visit(s) is s, f"identity broken on {type(s).__name__}"
