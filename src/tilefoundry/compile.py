"""Top-level ``tilefoundry.lower`` / ``tilefoundry.build`` / ``tilefoundry.compile`` entries.

Three public verbs, all accept ``Module`` exclusively.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass

from tilefoundry.inspection import as_script as _as_script
from tilefoundry.ir.core import Call, Expr, Tuple
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.passes.pass_manager import PassManager
from tilefoundry.passes.transforms import BufferizePass, HirToTirPass
from tilefoundry.schedule.constraints import AgentConstraintsMetadata


def _has_agent_constraints(expr: Expr | None) -> bool:
    if expr is None:
        return False
    if getattr(expr, "metadata", ()):
        if any(isinstance(item, AgentConstraintsMetadata) for item in expr.metadata):
            return True
    if isinstance(expr, Call):
        return any(_has_agent_constraints(argument) for argument in expr.args)
    if isinstance(expr, Tuple):
        return any(_has_agent_constraints(element) for element in expr.elements)
    return False

# ── Compiler Options ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompilerOptions:
    """Minimal compiler options for cache-key and compile configuration.

    Serialisation is deterministic: ``target`` + extra fields as
    sorted key-value pairs join by null separator.
    """
    target: str = "cuda"

    def canonical_text(self) -> str:
        """Deterministic text serialisation for cache-key computation."""
        parts = [f"target={self.target}"]
        return "\0".join(parts)

# ── Module Normalisation ─────────────────────────────────────────────────────

def normalize_to_module(fn_or_mod: HirFunction | Module) -> Module:
    """Normalise a ``Function`` or ``Module`` into a compile-ready ``Module``.

    - ``Function`` → single-function ``Module``; ``Function.topologies``
      are lifted into ``Module.topologies``.
    - ``Module`` → validated and returned as the compile unit.

    Raises ``TypeError`` for unsupported input types.
    """
    if isinstance(fn_or_mod, HirFunction):
        return Module(
            name=fn_or_mod.name,
            functions=(fn_or_mod,),
            entry=fn_or_mod.name,
            topologies=fn_or_mod.topologies,
        )
    if isinstance(fn_or_mod, Module):
        # Validate entry exists
        fn_or_mod.entry_function()  # raises ValueError if missing
        return fn_or_mod
    raise TypeError(
        f"normalize_to_module: expected Function or Module, "
        f"got {type(fn_or_mod).__name__}"
    )

def _build_default_pipeline() -> PassManager:
    pm = PassManager()
    pm.add(HirToTirPass())
    pm.add(BufferizePass())
    return pm

def lower(
    mod: Module,
    /,
    *,
    target: str = "cuda",
) -> Module:
    """Run the default pass pipeline on *mod* and return a lowered ``Module`` (TIR).

    *mod* must be a ``Module``.  Meshes are derived from the HIR body
    (``ShardLayout.mesh`` attributes on reshard ops), not from external
    parameters.
    """
    if not isinstance(mod, Module):
        raise TypeError(
            f"tilefoundry.lower: expected Module, got {type(mod).__name__}. "
            f"Construct Module(name=..., functions=(fn,), entry=fn.name) explicitly."
        )
    if target != "cuda":
        raise ValueError(f"tilefoundry.lower: target {target!r} not supported yet")

    for fn in mod.functions:
        if any(_has_agent_constraints(param) for param in getattr(fn, "params", ())):
            raise ValueError(
                "tilefoundry.lower: Module contains unresolved Agent Constraints; "
                "run tilefoundry.schedule.auto_dist first"
            )
        if _has_agent_constraints(getattr(fn, "body", None)):
            raise ValueError(
                "tilefoundry.lower: Module contains unresolved Agent Constraints; "
                "run tilefoundry.schedule.auto_dist first"
            )

    # Validate every declared program topology level against the target before
    # lowering — a function may declare an unsupported level (e.g. ``gpu``)
    # without ever emitting a MeshScope, so the codegen-side check is not enough.
    from tilefoundry.ir.target import validate_cuda_topology_levels  # noqa: PLC0415
    declared = list(mod.topologies)
    for fn in mod.functions:
        declared.extend(getattr(fn, "topologies", ()) or ())
    validate_cuda_topology_levels(t.name for t in declared)

    pm = _build_default_pipeline()
    result = pm.run(mod)
    merged = dict(result.metadata)
    merged["target"] = target
    return Module(
        name=result.name,
        functions=result.functions,
        entry=result.entry,
        topologies=result.topologies,
        metadata=merged,
    )

def build(
    mod: Module,
    /,
    *,
    target: str | None = None,
) -> "RuntimeModule":
    """Codegen + compile + load *mod* and return a fully-loaded ``RuntimeModule``.

    *mod* must be a ``Module``. *target* defaults to ``mod.metadata["target"]``.
    Raises ``ValueError`` if missing or if explicit *target* conflicts.
    """

    if not isinstance(mod, Module):
        raise TypeError(
            f"tilefoundry.build: expected Module, got {type(mod).__name__}."
        )
    mod_target: object = mod.metadata.get("target") if hasattr(mod, "metadata") else None
    if target is None:
        if mod_target is None:
            raise ValueError(
                "tilefoundry.build: target not specified and mod.metadata has no 'target'"
            )
        target = str(mod_target)
    elif mod_target is not None and target != str(mod_target):
        raise ValueError(
            f"tilefoundry.build: explicit target {target!r} conflicts with "
            f"mod.metadata['target'] = {mod_target!r}"
        )
    if target != "cuda":
        raise ValueError(f"tilefoundry.build: target {target!r} not supported yet")

    workdir = os.path.join(tempfile.gettempdir(), f"tilefoundry_build_{mod.entry}_split")
    return _build_split_runtime_module(mod, workdir=workdir)


def _build_split_runtime_module(mod: Module, *, workdir: str) -> "RuntimeModule":
    """Codegen + compile + load *mod* through the split host/device pipeline.

    All device-only modules route here: a CPU host entry is synthesized (or, for
    a dispatch entry, the entry is retargeted to CPU), each target emits its own
    ``LinkableModule``, and those modules are compiled separately and linked into
    one host-callable ``.so``. Unsupported module shapes raise during
    normalization / codegen — there is no fallback to a single-source path.
    """
    # noqa lazy: keep these heavy codegen/runtime imports off the module load
    # path and out of any import cycle with this top-level module.
    from tilefoundry.codegen.cuda.emit import (  # noqa: PLC0415
        _derive_launch_config,
        _output_count_from_fn,
        _param_abi,
    )
    from tilefoundry.codegen.cuda.tir.prim_function import (  # noqa: PLC0415
        _is_dispatch_entry_shape,
        _is_hidden_shape_scalar,
    )
    from tilefoundry.codegen.linker import link_modules  # noqa: PLC0415
    from tilefoundry.codegen.registry import (  # noqa: PLC0415
        get_emitter,
        group_functions_by_target,
    )
    from tilefoundry.passes.transforms.host_entry import (  # noqa: PLC0415
        insert_default_host_entry,
    )
    from tilefoundry.runtime.loader import load_linked_module  # noqa: PLC0415
    from tilefoundry.runtime.module import (  # noqa: PLC0415
        CallableType,
        KernelInfo,
        LaunchConfig,
    )

    linked = insert_default_host_entry(mod)
    groups = group_functions_by_target(linked)
    cuda_group = groups.get("cuda", ())
    if not cuda_group:
        raise ValueError(
            f"tilefoundry.build: module {linked.name!r} has no CUDA device functions"
        )
    cpu_entry = linked.entry_function()
    if cpu_entry.target.name != "cpu":
        raise ValueError(
            f"tilefoundry.build: entry {cpu_entry.name!r} is not a CPU host entry "
            f"after normalization"
        )

    device_module = get_emitter("cuda")(cuda_group)
    host_module = get_emitter("cpu")(cpu_entry, linked)

    # Host-visible ABI: filter the hidden shape scalars the host derives from
    # tensor shapes; keep the CPU entry's parameter order and output count.
    entry_buffer_params = tuple(
        p for p in cpu_entry.params if not _is_hidden_shape_scalar(p, cpu_entry.params)
    )
    entry_type = CallableType(
        name=cpu_entry.name,
        params=tuple(_param_abi(p) for p in entry_buffer_params),
        output_count=_output_count_from_fn(cpu_entry),
    )

    # Metadata lists every device kernel / dispatch variant (a dispatch entry
    # emits no kernel). All kernels share the device module's launch config
    # (enforced by the device module's topology check), so one LaunchConfig is
    # representative.
    kernel_fns = [f for f in cuda_group if not _is_dispatch_entry_shape(f)]
    kernels = tuple(
        KernelInfo(
            name=f"{f.name}_kernel",
            param_names=tuple(p.name for p in f.params),
        )
        for f in kernel_fns
    )
    # Metadata only — the split pipeline's real launch grid is computed in the
    # host wrapper. A dynamic (launch-provided) CTA extent yields grid.x=None
    # here rather than raising.
    grid, block = _derive_launch_config(kernel_fns[0].body)

    cuda_arch = cuda_group[0].target.arch.removeprefix("sm_")
    linked_module = link_modules(
        (device_module, host_module),
        workdir=workdir,
        lib_name=cpu_entry.name,
        entry=entry_type,
        launch_config=LaunchConfig(grid=grid, block=block),
        kernels=kernels,
        cuda_arch=cuda_arch,
    )
    return load_linked_module(linked_module)

def compile(
    mod: Module,
    /,
    *,
    target: str = "cuda",
) -> "RuntimeModule":
    """``build(lower(mod, target=target))`` — full compile shortcut.

    *mod* must be a ``Module``.  Meshes are derived from the IR body.
    """
    lowered = lower(mod, target=target)
    return build(lowered)

def _canonical_module_text(mod: Module) -> str:
    """Produce canonical text for cache-key: entry-function source + topologies.

    Includes ``Module.topologies`` in stable serialised form so that two
    modules with the same entry function but different topology
    declarations produce different cache keys.
    """
    fn_text = _as_script(mod.entry_function())
    # Append topology declarations in sorted-by-name stable form
    if mod.topologies:
        topo_lines = []
        for t in sorted(mod.topologies, key=lambda t: t.name):
            topo_lines.append(f"Topology({t.name!r}, {t.size})")
        fn_text += "\n" + "\n".join(topo_lines)
    return fn_text

def jit(
    fn_or_mod,
    /,
    *,
    target: str = "cuda",
    options: CompilerOptions | None = None,
    **kwargs,
) -> "RuntimeModule":
    """JIT-compile a ``hir.Function`` or ``Module`` and return a ``RuntimeModule``.

    ``Module`` is the compile unit.  ``Function`` input is normalised into a
    single-function ``Module`` and ``Function.topologies`` are lifted to
    ``Module.topologies`` before compiling.

    Cache key is ``sha256(canonical_module_text + target_text +
    canonical_options_text)`` — no Python object identity participates.

    Args:
        fn_or_mod: A ``hir.Function`` or ``Module``.
        target: Compilation target (currently only ``"cuda"``).
        options: Compiler options (defaults to ``CompilerOptions(target=target)``).

    Returns:
        A callable ``RuntimeModule``.

    Raises:
        TypeError: raw Python functions and unsupported types.
        ValueError: unrecognised targets.
    """
    # Reject all unexpected kwargs
    if kwargs:
        bad = ", ".join(kwargs.keys())
        raise TypeError(
            f"tilefoundry.jit: unexpected keyword argument(s): {bad}. "
            f"Accepted parameters are: fn_or_mod, target, options."
        )

    # Reject raw Python functions
    if not isinstance(fn_or_mod, (HirFunction, Module)):
        raise TypeError(
            f"tilefoundry.jit: expected Function or Module, "
            f"got {type(fn_or_mod).__name__}"
        )

    # Normalise
    mod = normalize_to_module(fn_or_mod)

    # Options
    if options is None:
        options = CompilerOptions(target=target)
    if target != "cuda":
        raise ValueError(f"tilefoundry.jit: target {target!r} not supported yet")

    # Cache key: sha256 over canonical module text + target + options
    canonical_text = _canonical_module_text(mod)
    payload = canonical_text + "\0" + target + "\0" + options.canonical_text()
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    if key not in _jit_cache:
        _jit_cache[key] = compile(mod, target=target)
    return _jit_cache[key]

_jit_cache: dict[str, "RuntimeModule"] = {}

def _jit_cache_clear() -> None:
    """Clear the jit cache (for testing)."""
    _jit_cache.clear()

def _jit_cache_info() -> dict:
    """Return cache stats dict."""
    return {"size": len(_jit_cache)}

jit.cache_clear = _jit_cache_clear  # type: ignore[attr-defined]
jit.cache_info = _jit_cache_info    # type: ignore[attr-defined]

def _jit_cache_key_payload(
    fn_or_mod: HirFunction | Module,
    target: str = "cuda",
    options: CompilerOptions | None = None,
) -> tuple[str, str, str]:
    """For testing only: return ``(module_text, target, options_text)``.

    The actual cache key is ``sha256(text + "\0" + target + "\0" + opts)``.
    """
    mod = normalize_to_module(fn_or_mod)
    if options is None:
        options = CompilerOptions(target=target)
    return (
        _canonical_module_text(mod),
        target,
        options.canonical_text(),
    )

__all__ = ["lower", "build", "compile", "jit", "normalize_to_module", "CompilerOptions"]
