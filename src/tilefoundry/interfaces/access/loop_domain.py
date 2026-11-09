"""The relation that constrains named induction values by authored loop bounds."""

from __future__ import annotations

import isl

from tilefoundry.ir import expr

from .relation import Relation, affine_of, param_name


def loop_domain(loops, values, *, first=False):
    """Constrain named induction values by the shared half-open loop bounds."""
    result = Relation(isl.map("{ [] -> [] }"))
    for induction_var, begin, end, stride in loops:
        start = affine_of(begin, values=values)
        stop = affine_of(end, values=values)
        step = expr.static_dim_value(stride)
        if step is None:
            raise ValueError("loop step must be static")
        name = param_name(induction_var)
        values[name] = induction_var
        induction = isl.pw_aff(f"[{name}] -> {{ [{name}] }}")
        if first:
            domain = start.eq_set(induction)
        else:
            domain = start.le_set(induction).intersect(induction.lt_set(stop))
            if step > 1:
                domain = domain.intersect(induction.sub(start).mod(isl.val(step)).zero_set())
        result = result.constrain(domain.params(), values)
    return result


__all__ = ["loop_domain"]
