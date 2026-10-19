"""Traversal environments prepare bindings; scopes answer analysis queries."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import isl

from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.utils import local_type_of
from tilefoundry.utils.isl_utils import has_unbounded_param
from tilefoundry.visitor_registry.access_relation import AccessRelation
from tilefoundry.visitor_registry.call_frame import CallFrame
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .errors import AnalysisError


@dataclass
class AnalyzeContext(TypeInferContext):
    options: object | None = None
    entry: Function | None = None
    scope: IterationScope | None = None
    definitions: dict[int, IterationScope] = field(default_factory=dict)
    caller_context: AnalyzeContext | None = field(default=None, repr=False)
    topology_level: str | None = None
    topologies: tuple = ()

    @property
    def module(self):
        return self.frame.module

    @property
    def target(self):
        return self.resolve_target()

    def type_of(self, expr):
        return expr.type

    def local_type_of(self, expr):
        if self.topology_level is None:
            return expr.type
        key = ("analysis_local", id(expr), self.topology_level, self.topologies)
        if key not in self.instantiated_memo:
            self.instantiated_memo[key] = local_type_of(
                expr.type, topology_level=self.topology_level, topologies=self.topologies
            )
        return self.instantiated_memo[key]

    def local_output_type(self, call):
        return self.local_type_of(call)

    def enter_region(self, region):
        inner = super().enter_region(region)
        return replace(
            inner, scope=next(child for child in self.scope.children if child.owner is region)
        )

    def enter_call(self, call):
        return replace(self, call_site=call)

    def for_callee(self, callee, *, frame=None):
        return replace(
            super().for_callee(callee, frame=frame),
            caller_context=self,
            scope=self.callee_scope if self.call_site is not None else self.definitions[id(callee)],
        )

    @property
    def callee_scope(self):
        return self.scope.attachment(self.call_site)

    def call_bindings(self, frame):
        """Prepare arguments and instance coordinates for the visitor's explicit attach."""
        formals = tuple(
            p
            for p in frame.function.params
            if isinstance(self.call_site.target, Function) or not p.is_const
        )
        arguments = {id(p): arg for p, arg in zip(formals, self.call_site.args, strict=True)}
        instance = None
        slot, index = CallFrame.module_slot(self.call_site)
        if slot is not None:
            values = {}
            try:
                relation = (
                    AccessRelation.affine(index, self.scope, values=values)
                    .insert_domain(isl.set("{ [] }").get_space())
                    .as_map()
                )
                instance = AccessRelation(relation, values).constrain(self.scope.domain)
                if has_unbounded_param(instance.relation):
                    raise ValueError("unbounded module index")
            except (TypeError, ValueError, NotImplementedError, isl.Error):
                count = len(self.module.children[slot.key].elements)
                instance = AccessRelation(
                    isl.map(f"{{ [] -> [i] : 0 <= i < {count} }}"), may_access=True
                )

        if slot is not None:
            offset = 0
            for key, child in self.module.children.items():
                if key == slot.key:
                    break
                offset += len(child.elements)
            instance = instance.compose(AccessRelation(isl.map(f"{{ [i] -> [i + {offset}] }}")))

        return arguments, instance

    def error(self, node, message):
        raise AnalysisError(message)


__all__ = ["AnalyzeContext"]
