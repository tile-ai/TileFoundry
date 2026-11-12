"""Shared lexical environment of every valuation visitor."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tilefoundry.ir import core, expr, types


@dataclass
class Context:
    """Shared lexical environment and call transitions for valuation visitors."""

    module: "Module | None" = None
    function: "Function | PrimFunction | None" = None
    target: "Target | None" = None
    topologies: "tuple[Topology, ...] | None" = None
    caller_context: Context | None = field(default=None, repr=False, compare=False)
    current_mesh: types.Mesh | None = None
    loops: tuple[tuple[expr.Expr, object, object, object], ...] = ()
    loop_aliases: dict[int, expr.Expr] = field(default_factory=dict, repr=False, compare=False)

    call_site: expr.Call | None = None
    owner: expr.Expr | None = None
    parent: Context | None = field(default=None, repr=False, compare=False)

    def with_captures(self, captures):
        """Keep captured induction parameters in their original loop coordinate."""
        aliases = dict(self.loop_aliases)
        for param, arg in captures:
            aliases[id(param)] = self.capture_root(arg)
        return replace(self, loop_aliases=aliases)

    def enter_region(self, region):
        """Enter a region after its arguments have been visited outside it."""
        inner = self.with_captures(region.captures())
        if isinstance(region, core.LoopRegion):
            loop = (region.induction_var, region.start, region.extent, region.step)
            return replace(inner, loops=(*self.loops, loop), owner=region, parent=self)
        mesh = types.Mesh.from_meshes(self.current_mesh, region.mesh) if self.current_mesh else region.mesh
        return replace(inner, current_mesh=mesh, owner=region, parent=self)

    def __post_init__(self):
        if self.module is not None:
            if self.target is None:
                self.target = self.module.target
            if self.topologies is None:
                self.topologies = tuple(self.module.topologies or ())

    def enter_call(self, call):
        """Keep operands in the caller environment while identifying this call."""
        return replace(self, call_site=call, caller_context=self)

    @property
    def callee(self):
        context = self.callee_context()
        return None if context is None else context.function

    def callee_context(self, call=None):
        """Prepare the called definition's environment without visiting its body."""
        call = self.call_site if call is None else call
        if call is None:
            return None
        if not isinstance(call.target, core.FunctionRef):
            return None
        return self.enter_reference(call.target)

    def enter_reference(self, ref):
        """The environment of the function *ref* names from this frame's Module.

        Each Module the path passes through may declare its own target and
        topologies; undeclared ones inherit from the Module before it.
        """
        module, function = core.resolve(self.module, ref)
        target, topologies = self.target, self.topologies
        for node in core.descend(self.module, ref):
            target = node.target if node.target is not None else target
            topologies = node.topologies if node.topologies is not None else topologies
        return replace(
            self,
            module=module,
            function=function,
            owner=function,
            target=target,
            topologies=topologies,
            call_site=None,
            parent=None,
            caller_context=self,
        )

    def capture_root(self, node):
        """Resolve captures without treating carried values as invariant aliases."""
        return self.loop_aliases.get(id(node), node)

    def ancestors(self):
        """Lexical regions, from the innermost context to the definition root."""
        current = self
        while current is not None:
            yield current
            current = current.parent

    def enclosing_loops(self):
        """Loop owners in their lexical order."""
        return tuple(
            reversed([ctx.owner for ctx in self.ancestors() if isinstance(ctx.owner, core.LoopRegion)])
        )

    def enclosing_mesh(self):
        """The nearest authored mesh, or the definition's external mesh."""
        return next(
            (ctx.owner.mesh for ctx in self.ancestors() if isinstance(ctx.owner, core.MeshRegion)),
            self.current_mesh,
        )

    def resolve_target(self):
        """The target carried by this visitor's current environment."""
        return self.target


__all__ = ["Context"]
