"""The relation one boundary of an Op states, with what its parameters are."""

from __future__ import annotations

from dataclasses import dataclass, field

import isl

from tilefoundry.ir import dim


@dataclass(frozen=True)
class Relation:
    """One boundary's relation, together with what its parameters are.

    A coordinate an Op only learns at run time is a parameter rather than a hole:
    *values* maps each parameter's name in *relation* to the operand element or
    dimension it is, so whoever restricts the relation binds it rather than
    guessing. A relation with no parameters states none, and a function handed in
    is kept as the relation it is. How much crossed here is what the relation
    reaches, so it is derived rather than declared alongside.
    """

    relation: "isl.map"
    values: dim.IslParamValues = field(default_factory=dict)
    may_access: bool = False
    affine: isl.multi_aff | None = None

    def __post_init__(self) -> None:
        if isinstance(self.relation, isl.multi_aff):
            object.__setattr__(self, "relation", isl.map.from_multi_aff(self.relation))
        if not isinstance(self.relation, isl.map):
            raise ValueError(f"an access relation is a relation, not {self.relation!r}")
        object.__setattr__(self, "values", dict(self.values))
        named = set(parameter_names(self.relation))
        if named != set(self.values):
            raise ValueError(
                f"an access relation binds {sorted(self.values)} but its relation names "
                f"{sorted(named)}; a parameter nobody can bind is a hole"
            )

    def aligned(self, relation, values=(), *, may_access=False):
        """Keep the parameter table exactly aligned with an operation's result."""
        values = {**self.values, **dict(values)}
        return Relation(
            relation,
            {name: values[name] for name in parameter_names(relation)},
            self.may_access or may_access,
        )

    def compose(self, other: Relation) -> Relation:
        return self.aligned(
            self.relation.apply_range(other.relation), other.values, may_access=other.may_access
        )

    def constrain(self, params, values=()) -> Relation:
        if isinstance(params, Relation):
            values, params = params.values, params.relation.params()
        if not params.dim(isl.dim_type.PARAM) and params.plain_is_universe():
            return self
        return self.aligned(self.relation.intersect_params(params), values)

    def project(self, exprs, ctx=None) -> Relation:
        names = {param_name(node, ctx) for node in exprs}
        if not names.intersection(self.values):
            return self
        relation = self.relation
        for index in reversed(range(relation.dim(isl.dim_type.PARAM))):
            if relation.get_dim_name(isl.dim_type.PARAM, index) in names:
                relation = relation.project_out(isl.dim_type.PARAM, index, 1)
        return self.aligned(relation)

    def normalized(self, ctx=None) -> Relation:
        """Translate an op's local parameter spelling once at the analysis boundary."""
        if not self.values:
            return self
        relation = self.relation
        values = {}
        for name, value in self.values.items():
            value = ctx.capture_root(value) if ctx is not None else value
            target = param_name(value)
            index = relation.find_dim_by_name(isl.dim_type.PARAM, name)
            other = relation.find_dim_by_name(isl.dim_type.PARAM, target)
            if index < 0:
                continue
            if other >= 0 and other != index:
                relation = relation.equate(isl.dim_type.PARAM, index, isl.dim_type.PARAM, other)
                relation = relation.project_out(isl.dim_type.PARAM, index, 1)
            else:
                relation = relation.set_dim_name(isl.dim_type.PARAM, index, target)
            values[target] = value
        return Relation(relation, values, self.may_access)


def parameter_names(relation) -> list[str]:
    """The parameters *relation* names, in its own order."""
    return [
        relation.get_dim_name(isl.dim_type.PARAM, index)
        for index in range(relation.dim(isl.dim_type.PARAM))
    ]


def param_name(node, ctx=None) -> str:
    """The parameter name one value takes, the same for every relation naming it."""
    return f"__tf_{id(ctx.capture_root(node) if ctx is not None else node)}"


def affine_of(node, ctx=None, *, values=None):
    """One dimension expression as an isl piecewise affine function of its parameters."""
    return dim.dimension_to_isl_pw_aff(
        node,
        {} if values is None else values,
        param_name=lambda value: param_name(value, ctx),
    )


__all__ = ["Relation", "affine_of", "param_name", "parameter_names"]
