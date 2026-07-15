"""Forward access-relation construction helpers (input-type driven)."""
from __future__ import annotations

import isl

from tilefoundry.visitor_registry.isl_utility import to_dim, to_domain


def build_domain(extents: tuple) -> "isl.set":
    """Bounded iteration domain for *extents* (see ``isl_utility.to_domain``).
    A caller that also needs the output shape back (``shape_from_relation``)
    should call ``to_domain`` directly instead, to keep its param map
    alongside the relation."""
    domain, _ = to_domain(extents)
    return domain


def shape_from_relation(relation) -> tuple:
    """Derive the output shape from the relation's output map + bounded domain.

    Each output map result axis is a pure projection of a domain dim (its
    extent is recovered via ``isl_utility.to_dim`` and ``relation.param_map``)
    or a constant (a size-1 output axis). Anything else fails closed.
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
            shape.append(1)
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
