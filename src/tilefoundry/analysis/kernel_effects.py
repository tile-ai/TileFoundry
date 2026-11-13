"""Derive external byte effects from verified local-memory kernels."""

from dataclasses import replace

import isl

from tilefoundry import services
from tilefoundry.interfaces import access, execution, verify
from tilefoundry.ir import core, dim, expr, types
from tilefoundry.ir.tir import memory, sync


class Collector(core.StmtVisitor):
    def __init__(self, function):
        self.params = function.params
        self.context = verify.Context(function=function)
        self.bindings = {id(param): (index, access.byte_relation(param.type), 0)
                         for index, param in enumerate(function.params) if isinstance(param.type, types.TensorType)}
        self.defined = {}
        self.effects = []
        self.pending = []
        self.return_allowed = True

    def resolve_capture(self, value):
        """Substitute static captures throughout a dimension expression."""
        if isinstance(value, expr.Call) and isinstance(value.target, expr.DIM_OP_TYPES):
            cloner = core.BindingSubstitutionCloner()
            return services.normalize_dim(cloner.visit(value, self.context.loop_aliases))
        return self.context.capture_root(value)

    def resolved(self, relation):
        result, may = relation.relation, relation.may_access
        for name, value in relation.values.items():
            value = self.resolve_capture(value)
            number = expr.static_dim_value(value)
            axis = result.find_dim_by_name(isl.dim_type.PARAM, name)
            if number is None:
                may = True
                self.control_read(value)
            else:
                result = result.fix_val(isl.dim_type.PARAM, axis, isl.val(number))
            result = result.project_out(isl.dim_type.PARAM, axis, 1)
        if not result.range().is_bounded():
            raise ValueError("kernel accesses require bounded byte regions")
        return access.Relation(result, may_access=may)

    def control_read(self, value):
        """Retain external scalar inputs that determine a view's addresses."""
        if isinstance(value, expr.Var):
            origin = self.bindings.get(id(value))
            if origin is not None:
                index, relation, _ = origin
                self.pending.append(execution.Effect(index, relation, expr.MemoryEffect.READ))
            return
        if isinstance(value, expr.Call) and isinstance(value.target, expr.DIM_OP_TYPES):
            for arg in value.args:
                self.control_read(arg)

    def first_byte(self, region, width, *, required):
        first = region.relation
        for axis in range(first.dim(isl.dim_type.IN)):
            first = first.fix_val(isl.dim_type.IN, axis, isl.val(0))
        first_bytes = first.range()
        low = first_bytes.dim_min_val(0)
        base = low.get_num_si() if low.is_int() else None
        if base is not None and not first_bytes.is_equal(isl.set(f"{{ [b] : {base} <= b < {base + width} }}")):
            base = None
        if required and base is None:
            raise ValueError("kernel pointer views require a unique known first byte")
        return base

    def origin(self, value):
        if isinstance(value, expr.Var):
            if id(value) not in self.bindings:
                raise ValueError(f"unbound kernel operand: {value.name}")
            return self.bindings[id(value)]
        if isinstance(value, (expr.Constant, core.ShapeOf)):
            return None
        if not isinstance(value, expr.Call):
            raise ValueError(f"unmodeled kernel operand: {type(value).__name__}")
        if isinstance(value.target, memory.AllocTensor):
            return None
        if isinstance(value.target, memory.PtrOf):
            return self.origin(value.args[0])
        if isinstance(value.target, memory.TensorView):
            origin = self.origin(value.args[0])
            if origin is None:
                return None
            index, _, base = origin
            if base is None:
                raise ValueError("kernel pointer views require a unique known first byte")
            region = self.resolved(access.byte_relation(value.type)).compose(access.Relation(isl.map(f"{{ [b] -> [c] : c = b + {base} }}")))
            base = self.first_byte(region, value.type.dtype.bit_width // 8, required=True)
            return index, region, base
        position = access.aliased_operand(value)
        if position is not None:
            origin = self.origin(value.args[position])
            if origin is None:
                return None
            index, parent, _ = origin
            alias = access.renaming_relation(value, self.context, access.relations_of(value, self.context))
            region = self.resolved(alias).compose(parent)
            base = self.first_byte(region, value.type.dtype.bit_width // 8, required=False)
            return index, region, base
        raise ValueError(f"unmodeled kernel alias or expression: {type(value.target).__name__}")

    def visit_LetStmt(self, stmt):
        self.bindings[id(stmt.var)] = self.origin(stmt.value)

    def visit_Sequential(self, stmt):
        for index, child in enumerate(stmt.body):
            if isinstance(child, core.Return) and (index + 1 < len(stmt.body) or not self.return_allowed):
                raise ValueError("kernel effect summary does not support early returns")
            self.visit(child)

    def visit_Evaluate(self, stmt):
        if isinstance(stmt.callable, sync.Sync) and not stmt.args:
            return
        if not isinstance(stmt.callable, expr.Op) or access.relation_registry.get(type(stmt.callable), None) is None:
            raise ValueError(f"unmodeled kernel effect: {type(stmt.callable).__name__}")
        params = tuple(param for param in expr.signature_of(type(stmt.callable)) if param.kind == "input")
        if len(params) != len(stmt.args) or any(param.effect is None for param in params):
            raise ValueError("kernel effect summary requires bound operands with declared effects")
        call = expr.Call(stmt.callable, stmt.args, type=types.UnitType())
        relations = access.relations_of(call, self.context)
        self.pending = []
        for param, arg, relation in zip(params, stmt.args, relations[:len(params)], strict=True):
            origin = self.origin(arg)
            if origin is None:
                continue
            index, mapping, _ = origin
            region = self.resolved(relation).compose(mapping)
            reached = region.relation.range()
            allowed = access.byte_relation(self.params[index].type).relation.range()
            if not reached.is_subset(allowed):
                raise ValueError(f"kernel access exceeds parameter {self.params[index].name!r}")
            if not reached.is_empty():
                self.pending.append(execution.Effect(index, region, param.effect))
        for effect in self.pending:
            if effect.mode & expr.MemoryEffect.READ:
                needed = effect.region.relation.range()
                if effect.parameter in self.defined:
                    needed = needed.subtract(self.defined[effect.parameter])
                if not needed.is_empty():
                    self.effects.append(replace(effect, mode=expr.MemoryEffect.READ,
                                                region=access.Relation(isl.map.from_range(needed), may_access=effect.region.may_access)))
        for effect in self.pending:
            if effect.mode & expr.MemoryEffect.WRITE:
                self.effects.append(replace(effect, mode=expr.MemoryEffect.WRITE))
                if not effect.region.may_access:
                    reached = effect.region.relation.range()
                    previous = self.defined.get(effect.parameter)
                    self.defined[effect.parameter] = reached if previous is None else previous.union(reached)

    def visit_Return(self, stmt):
        pass

    def visit_If(self, stmt):
        raise ValueError("kernel effect summary does not yet merge conditional aliases")

    def visit_While(self, stmt):
        raise ValueError("kernel effect summary requires static control flow")

    def visit_MeshScope(self, stmt):
        previous = self.context
        self.context = replace(previous)
        self.context.enter_scope((*previous.mesh_scope, stmt.mesh))
        self.visit(stmt.body)
        self.context = previous

    def visit_For(self, stmt):
        bounds = tuple(expr.static_dim_value(value) for value in (stmt.start, stmt.stop, stmt.step))
        if any(value is None for value in bounds) or bounds[2] == 0:
            raise ValueError("kernel effect summary requires static nonzero-step loops")
        context = self.context
        allowed = self.return_allowed
        self.return_allowed = False
        for value in range(*bounds):
            self.context = context.with_captures(((stmt.induction_var, dim.i64_const(value)),))
            self.bindings[id(stmt.induction_var)] = None
            self.visit(stmt.body)
        self.context = context
        self.return_allowed = allowed


def summarize(function: core.PrimFunction) -> execution.Summary:
    """Summarize external local accesses that finish at kernel completion."""
    collector = Collector(function)
    collector.visit(function.body)
    merged = {}
    for effect in collector.effects:
        key = effect.parameter, effect.mode, effect.region.may_access
        reached = effect.region.relation.range()
        merged[key] = reached if key not in merged else merged[key].union(reached)
    return execution.Summary(tuple(execution.Effect(parameter, access.Relation(isl.map.from_range(region), may_access=may), mode)
                                   for (parameter, mode, may), region in merged.items()))


__all__ = ["summarize"]
