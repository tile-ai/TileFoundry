"""Evaluate and solve the formulas of operation-declaration patterns.

A formula is read through its ``op`` and ``args`` alone. Evaluation answers
true, false or ``None`` while a named value is unbound; solving hands what
remains to CP-SAT.
"""

from __future__ import annotations

from itertools import count

from tilefoundry import utils

cp_model = utils.LazyModule("ortools.sat.python.cp_model")

MISSING = object()


class Unknown:
    """A truthy result when the solver cannot prove success or failure."""

    def __bool__(self) -> bool:
        return True


UNKNOWN = Unknown()


def value_of(term: Term, env: dict, local: dict | None = None):
    local = {} if local is None else local
    if term.op == "constant":
        return term.args[0]
    if term.op == "variable":
        return env.get(term.args[0], MISSING)
    if term.op == "bound":
        return local.get(term.args[0], MISSING)
    if term.op == "bits":
        value = env.get(term.args[0], MISSING)
        return MISSING if value is MISSING else getattr(value, "bit_width", MISSING)
    if term.op == "element":
        values, index = term.args
        index = value_of(index, env, local)
        return MISSING if index is MISSING else values[index]
    if term.op in {"sum", "count"}:
        value = value_of(term.args[0], env, local)
        if value is MISSING:
            return MISSING
        if not isinstance(value, tuple):
            raise TypeError(f"{term.op.title()} requires a tuple binding")
        return sum(value) if term.op == "sum" else len(value)
    left = value_of(term.args[0], env, local)
    right = value_of(term.args[1], env, local)
    if left is MISSING or right is MISSING:
        return MISSING
    return {
        "add": lambda: left + right,
        "sub": lambda: left - right,
        "mul": lambda: left * right,
        "floordiv": lambda: left // right,
        "mod": lambda: left % right,
    }[term.op]()


def truth_of(formula: Formula, env: dict, local: dict | None = None) -> bool | None:
    local = {} if local is None else local
    if formula.op in {"and", "or"}:
        left = truth_of(formula.args[0], env, local)
        right = truth_of(formula.args[1], env, local)
        if formula.op == "and":
            return False if False in (left, right) else None if None in (left, right) else True
        return True if True in (left, right) else None if None in (left, right) else False
    if formula.op == "not":
        value = truth_of(formula.args[0], env, local)
        return None if value is None else not value
    if formula.op == "forall":
        term, token, body = formula.args
        values = value_of(term, env, local)
        if values is MISSING:
            raise TypeError(
                "ForAll requires a tuple whose length is known after structural matching"
            )
        if not isinstance(values, tuple):
            raise TypeError("ForAll requires a tuple binding")
        found = tuple(truth_of(body, env, {**local, token: value}) for value in values)
        return False if False in found else None if None in found else True
    if formula.op == "in":
        value = value_of(formula.args[0], env, local)
        return None if value is MISSING else value in formula.args[1]
    left = value_of(formula.args[0], env, local)
    right = value_of(formula.args[1], env, local)
    if left is MISSING or right is MISSING:
        return None
    return {
        "eq": lambda: left == right,
        "ne": lambda: left != right,
        "lt": lambda: left < right,
        "le": lambda: left <= right,
        "gt": lambda: left > right,
        "ge": lambda: left >= right,
    }[formula.op]()


def evaluate(formula: Formula, env) -> bool | None:
    """Evaluate *formula*, returning None when a named value is unbound."""
    return truth_of(formula, dict(env or {}))


class CpBuilder:
    """Compile the unresolved portion of formulas into one CP-SAT model."""

    LIMIT = 2**31 - 1

    def __init__(self, model, env):
        self.model = model
        self.env = dict(env)
        self.variables = {}
        self.serial = count()

    def fresh(self, prefix: str):
        return self.model.new_int_var(-self.LIMIT, self.LIMIT, f"_{prefix}{next(self.serial)}")

    def term(self, term: Term, local=None):
        local = {} if local is None else local
        known = value_of(term, self.env, local)
        if known is not MISSING:
            if type(known) is not int:
                raise TypeError(f"CP-SAT arithmetic requires integers, got {known!r}")
            return known
        if term.op == "variable":
            name = term.args[0]
            if name not in self.variables:
                self.variables[name] = self.model.new_int_var(-self.LIMIT, self.LIMIT, name)
            return self.variables[name]
        if term.op == "bits":
            raise TypeError(f"Bits({term.args[0]!r}) requires its dtype binding")
        if term.op == "bound":
            raise TypeError("ForAll requires its tuple binding before CP-SAT solving")
        if term.op == "element":
            values, index = term.args
            target = self.fresh("element")
            self.model.add_element(self.term(index, local), values, target)
            return target
        if term.op in {"sum", "count"}:
            raise TypeError(f"{term.op.title()} requires its tuple binding before CP-SAT solving")
        left = self.term(term.args[0], local)
        right = self.term(term.args[1], local)
        if term.op == "add":
            return left + right
        if term.op == "sub":
            return left - right
        target = self.fresh(term.op)
        if term.op == "mul":
            self.model.add_multiplication_equality(target, (left, right))
        elif term.op == "floordiv":
            self.model.add_division_equality(target, left, right)
        elif term.op == "mod":
            self.model.add_modulo_equality(target, left, right)
        else:
            raise ValueError(f"unknown Term operation {term.op!r}")
        return target

    def formula(self, formula: Formula, local=None):
        local = {} if local is None else local
        known = truth_of(formula, self.env, local)
        if known is not None:
            return self.model.new_constant(known)
        if formula.op == "not":
            return self.formula(formula.args[0], local).Not()
        if formula.op in {"and", "or"}:
            parts = tuple(self.formula(part, local) for part in formula.args)
            result = self.model.new_bool_var(f"_{formula.op}{next(self.serial)}")
            if formula.op == "and":
                self.model.add_bool_and(parts).only_enforce_if(result)
                self.model.add_bool_or(tuple(part.Not() for part in parts)).only_enforce_if(
                    result.Not()
                )
            else:
                self.model.add_bool_or(parts).only_enforce_if(result)
                self.model.add_bool_and(tuple(part.Not() for part in parts)).only_enforce_if(
                    result.Not()
                )
            return result
        if formula.op == "forall":
            term, token, body = formula.args
            values = value_of(term, self.env, local)
            if values is MISSING:
                raise TypeError(
                    "ForAll requires a tuple whose length is known after structural matching"
                )
            if not isinstance(values, tuple):
                raise TypeError("ForAll requires a tuple binding")
            parts = tuple(self.formula(body, {**local, token: value}) for value in values)
            result = self.model.new_bool_var(f"_forall{next(self.serial)}")
            self.model.add_bool_and(parts).only_enforce_if(result)
            self.model.add_bool_or(tuple(part.Not() for part in parts)).only_enforce_if(
                result.Not()
            )
            return result
        result = self.model.new_bool_var(f"_{formula.op}{next(self.serial)}")
        if formula.op == "in":
            value = self.term(formula.args[0], local)
            rows = tuple((item,) for item in formula.args[1])
            self.model.add_allowed_assignments((value,), rows).only_enforce_if(result)
            self.model.add_forbidden_assignments((value,), rows).only_enforce_if(result.Not())
            return result
        left = self.term(formula.args[0], local)
        right = self.term(formula.args[1], local)
        relation = {
            "eq": lambda: left == right,
            "ne": lambda: left != right,
            "lt": lambda: left < right,
            "le": lambda: left <= right,
            "gt": lambda: left > right,
            "ge": lambda: left >= right,
        }[formula.op]
        opposite = {
            "eq": lambda: left != right,
            "ne": lambda: left == right,
            "lt": lambda: left >= right,
            "le": lambda: left > right,
            "gt": lambda: left <= right,
            "ge": lambda: left < right,
        }[formula.op]
        self.model.add(relation()).only_enforce_if(result)
        self.model.add(opposite()).only_enforce_if(result.Not())
        return result


def solve(formulas, env) -> dict | None | Unknown:
    """Solve unresolved formulas, or return UNKNOWN without treating it as failure."""
    formulas = tuple(formulas)
    held = dict(env or {})
    evaluated = tuple(evaluate(formula, held) for formula in formulas)
    if False in evaluated:
        return None
    unresolved = tuple(formula for formula, value in zip(formulas, evaluated) if value is None)
    if not unresolved:
        return held

    model = cp_model.CpModel()
    builder = CpBuilder(model, held)
    for formula in unresolved:
        model.add(builder.formula(formula) == 1)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status == cp_model.UNKNOWN:
        return UNKNOWN
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return None
    held.update({name: solver.value(value) for name, value in builder.variables.items()})
    return held


def failing(formulas, env) -> Formula | None:
    """Return the first false conjunct, if evaluation can identify one."""
    for formula in formulas:
        if formula.op == "and":
            found = failing(formula.args, env)
            if found is not None:
                return found
        elif evaluate(formula, env) is False:
            return formula
    return None


__all__ = ["UNKNOWN", "Unknown", "evaluate", "failing", "solve"]
