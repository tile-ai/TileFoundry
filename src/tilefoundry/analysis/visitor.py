"""Traversal environments prepare bindings; scopes answer analysis queries."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import isl

from tilefoundry import utils
from tilefoundry.interfaces import access, typeinfer, valuation
from tilefoundry.ir import core
from tilefoundry.ir import utils as ir_utils

from .errors import Error


@dataclass
class AnalyzeContext(typeinfer.TypeInferContext):
    options: object | None = None
    entry: core.Function | None = None
    scope: IterationScope | None = None
    definitions: dict[int, IterationScope] = field(default_factory=dict)
    caller_context: AnalyzeContext | None = field(default=None, repr=False)
    topology_level: str | None = None

    def bind_scope(self, scope):
        """Make *scope* the one this context answers queries from."""
        self.scope = scope

    def type_of(self, expr):
        return expr.type

    def local_type_of(self, expr):
        if self.topology_level is None:
            return expr.type
        key = ("analysis_local", id(expr), self.topology_level, self.topologies)
        if key not in self.instantiated_memo:
            self.instantiated_memo[key] = ir_utils.local_type_of(
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
        inner = super().enter_call(call)
        frame = inner.callee_context()
        if frame is not None:
            arguments, instance = inner.call_bindings(frame)
            self.scope.attach(
                call_site=call,
                callee=self.definitions[id(frame.function)],
                arguments=arguments,
                instance=instance,
            )
        return inner

    @property
    def callee_scope(self):
        return self.scope.attachment(self.call_site)

    def call_bindings(self, frame):
        """Prepare arguments and instance coordinates for the visitor's explicit attach."""
        formals = tuple(
            p
            for p in frame.function.params
            if isinstance(self.call_site.target, core.Function) or not p.is_const
        )
        arguments = {id(p): arg for p, arg in zip(formals, self.call_site.args, strict=True)}
        instance = None
        slot, index = valuation.module_slot(self.call_site)
        if slot is not None:
            values = {}
            try:
                relation = (
                    access.affine_of(index, self.scope, values=values)
                    .insert_domain(isl.set("{ [] }").get_space())
                    .as_map()
                )
                instance = access.Relation(relation, values).constrain(self.scope.domain)
                if utils.has_unbounded_param(instance.relation):
                    raise ValueError("unbounded module index")
            except (TypeError, ValueError, NotImplementedError, isl.Error):
                count = len(self.module.children[slot.key].elements)
                instance = access.Relation(
                    isl.map(f"{{ [] -> [i] : 0 <= i < {count} }}"), may_access=True
                )

        if slot is not None:
            offset = 0
            for key, child in self.module.children.items():
                if key == slot.key:
                    break
                offset += len(child.elements)
            instance = instance.compose(access.Relation(isl.map(f"{{ [i] -> [i + {offset}] }}")))

        return arguments, instance

    def error(self, node, message):
        raise Error(message)


__all__ = ["AnalyzeContext"]
