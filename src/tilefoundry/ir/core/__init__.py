"""Structure nodes, the structure-aware visitors and the printing contract of the IR."""

from __future__ import annotations

from .evaluate import Evaluate
from .for_stmt import For
from .function import Function
from .function_ref import FunctionRef, is_index_type
from .if_stmt import If
from .inspection import Printable, PrinterBase
from .let_stmt import LetStmt
from .loop_region import LoopRegion
from .mesh_region import MeshRegion
from .mesh_scope import MeshScope
from .module import LoadedModule, Module, ModuleFunction, ModuleList, descend, instances, resolve
from .prim_function import PrimFunction
from .return_stmt import Return
from .sequential import Sequential
from .shape_of import ShapeOf, shape_of
from .ssa_value import SSAValue, ssa_leaves
from .stmt import Stmt
from .values import TotalAndPerUnit, TripInterval
from .visitor import (
    BindingSubstitutionCloner,
    ExprCloner,
    ExprCollector,
    ExprVisitor,
    ExprWalker,
    StmtExprMutator,
    StmtMutator,
    StmtVisitor,
    TypeFunctor,
    collect_exprs,
    expr_children,
    function_values,
    rewrite_prim_function,
    walk_prim_function,
)
from .while_stmt import While
