"""Forward access-relation construction helpers (input-type driven).

Builds the bounded iteration ``domain`` (an ``isl.set``) for an op from its
iteration extents, and recovers an output shape from a built relation. Both
directions are a thin consumer of ``tilefoundry.utilities.isl_utility`` (the
``ShapeDim`` <-> isl bridge) — this module owns no ``ShapeDim`` <-> isl
translation of its own.
"""
from __future__ import annotations

import isl

from tilefoundry.utilities.isl_utility import to_dim, to_domain


def build_domain(extents: tuple) -> "isl.set":
    """Bounded iteration domain ``{ [d0, ..., dn] : 0 <= di < extent_i }``
    (see ``isl_utility.to_domain``). For a relation whose output shape will
    be recovered via ``shape_from_relation``, call ``to_domain`` directly
    instead, so its ``param_map`` travels alongside the relation."""
    return to_domain(extents).domain


def shape_from_relation(relation) -> tuple:
    """Derive the output shape from the relation's output map + bounded domain.

    Each output map result axis is a pure projection of a domain dim — its
    extent (``domain.dim_max(d) + 1``) is recovered as a ``ShapeDim`` via
    ``isl_utility.to_dim`` and ``relation.param_map`` — or a constant (a
    size-1 output axis). A non-projection / non-constant result axis, or an
    extent that does not resolve to a ShapeDim, fails closed.
    """
    domain = relation.domain
    output_map = relation.maps[-1]
    ma = output_map.as_pw_multi_aff().as_multi_aff()
    n_out = ma.dim(isl.dim_type.OUT)
    n_in = ma.dim(isl.dim_type.IN)
    shape: list = []
    for o in range(n_out):
        aff = ma.get_at(o)
        used = [
            (j, int(aff.get_coefficient_val(isl.dim_type.IN, j).num_si()))
            for j in range(n_in)
            if int(aff.get_coefficient_val(isl.dim_type.IN, j).num_si()) != 0
        ]
        if not used:
            shape.append(1)  # constant result: a size-1 output axis
        elif len(used) == 1 and used[0][1] == 1:
            extent = domain.dim_max(used[0][0]).add_constant(1)
            shape.append(to_dim(extent, relation.param_map))
        else:
            raise ValueError(
                f"output axis {o} is not a pure projection or constant; "
                "cannot infer shape"
            )
    return tuple(shape)


def validate_output_map_arity(output_map: "isl.map", output_shape: tuple) -> None:
    """Check the output access map's range rank matches the claimed output
    shape rank. The relation carries no shape, so this is the consistency
    point between the relation and the typeinfer-side output shape."""
    n_out = output_map.dim(isl.dim_type.OUT)
    if n_out != len(output_shape):
        raise ValueError(
            f"output map range rank {n_out} != output shape rank {len(output_shape)}"
        )


__all__ = ["build_domain", "shape_from_relation", "validate_output_map_arity"]
