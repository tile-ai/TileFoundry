"""Service functions declared here and implemented by higher layers.

A lower layer that needs an implementation living above it calls one of these
functions. The implementing package registers its implementation at import,
keyed by the function; a call with no implementation raises
``MissingHandlerError``. This module imports no IR.
"""

from __future__ import annotations

from tilefoundry import dispatch

__all__ = [
    "builtin_analyzer",
    "describe_refusal",
    "dim_range",
    "evaluate",
    "normalize_dim",
    "prepare",
    "refusal_reason",
    "registry",
    "render",
    "report_json",
    "simplify_dim",
    "specialize",
    "converter",
    "verify_call",
]

registry = dispatch.Registry("services")


def evaluate(target: object, *args: object, context: object | None = None) -> object:
    """Evaluate a Function, Module or loaded Module on the reference interpreter."""
    return registry.require(evaluate)(target, *args, context=context)


def prepare(module: object, raw: object, out_dir: str, *, device: str = "cpu") -> None:
    """Prepare a Module's canonical weights and write them to *out_dir*."""
    return registry.require(prepare)(module, raw, out_dir, device=device)


def render(expr: object) -> str:
    """Render an IR expression as TIR text."""
    return registry.require(render)(expr)


def report_json(result: object) -> str:
    """Render the report data of one analysis result as JSON text."""
    return registry.require(report_json)(result)


def verify_call(op: object, args: tuple, mesh: object, *, frame: object) -> None:
    """Verify one scheduled TIR call; raise ``VerifyError`` when it is refused."""
    return registry.require(verify_call)(op, args, mesh, frame=frame)


def describe_refusal(instruction: str, path: tuple, reason: object) -> str:
    """Text of a schedule refusal of *instruction* at the binding *path*."""
    return registry.require(describe_refusal)(instruction, path, reason)


def refusal_reason(refusal: object, parameter: str | None) -> object:
    """The reason a failed pattern match gives for *parameter*."""
    return registry.require(refusal_reason)(refusal, parameter)


def builtin_analyzer(selector: str) -> object:
    """The standard analysis service named by *selector*, or ``None`` when there is none."""
    return registry.require(builtin_analyzer)(selector)


def simplify_dim(kind: str, args: tuple) -> object:
    """Build the dimension arithmetic *kind* over *args*, without applying algebraic rules.

    *kind* is one of ``"add"``, ``"sub"``, ``"mul"``, ``"floordiv"``, ``"mod"``,
    ``"min"`` and ``"max"``. Raw integers become i64 constants and bool is refused.
    """
    return registry.require(simplify_dim)(kind, args)


def normalize_dim(value: object) -> object:
    """The isl affine normal form of every dimension in *value*.

    Tuples are normalized entry-wise. A value that is not a dimension is returned
    unchanged, and so is *value* itself when no entry changed.
    """
    return registry.require(normalize_dim)(value)


def dim_range(value: object) -> tuple[int, int] | None:
    """Conservative half-open value bounds ``[lo, hi)`` of the dimension *value*, if known."""
    return registry.require(dim_range)(value)


def specialize(base: object, declared: object, mesh: object) -> object:
    """The decorator that registers a shape variant of the Function *base* over *declared*."""
    return registry.require(specialize)(base, declared, mesh=mesh)


def converter(base: object, weight_name: str) -> object:
    """The decorator that registers a per-weight offline converter of the Function *base*."""
    return registry.require(converter)(base, weight_name)
