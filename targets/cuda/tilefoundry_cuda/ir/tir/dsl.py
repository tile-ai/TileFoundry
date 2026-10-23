"""``T.cuda`` — CUDA instruction declarations."""

from __future__ import annotations

import enum
from typing import Callable

from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_namespace
from tilefoundry.ir.types import DType
from tilefoundry_cuda.ir.tir.nn.mma_atom import MmaAtom
from tilefoundry_cuda.ir.tir.nn.sm80_mma import Mma
from tilefoundry_cuda.ir.tir.nn.sm89_mma import Mma as Sm89Mma
from tilefoundry_cuda.ir.tir.nn.wgmma import Form, Major, Wgmma


class _Sm80Namespace:
    Mma = Mma


class _Sm89Namespace:
    Mma = Sm89Mma


class _Sm90Namespace:
    Wgmma = Wgmma
    Form = Form
    Major = Major


class _CudaNamespace:
    """``T.cuda`` — compile-time CUDA instruction declarations."""

    sm80 = _Sm80Namespace()
    sm89 = _Sm89Namespace()
    sm90 = _Sm90Namespace()


cuda = _CudaNamespace()


def _stub(annotation: Callable[[ParamDef], str]) -> str:
    """Typed stubs for declarative ``T.cuda.sm80`` / ``T.cuda.sm90`` atoms.

    These are compile-time descriptor surfaces, not OpSchema-backed ops, so the
    schema walk never sees them. Each constructor signature is reflected from
    the declaration's ``ParamDef`` fields. A ``DType`` parameter reuses the
    schema walk's annotation renderer, ``Literal[<dtype names>] | DType``,
    because the atom resolves a dtype name at construction; which dtypes it
    takes is still its pattern's answer. Other Enum parameters stay their Enum
    type, since no string spelling of them is accepted.
    """
    def constructor(name: str, declaration: type[MmaAtom]) -> str:
        params = []
        for param in declaration.parameters:
            if param.annotation is DType:
                param_type = annotation(param)
            else:
                param_type = getattr(param.annotation, "__name__", "Any")
            rendered = f"{param.name}: {param_type}"
            if param.has_default:
                rendered += " = ..."
            params.append(rendered)
        params.append("mesh: Mesh | None = ...")
        return (
            "    @staticmethod\n"
            f"    def {name}(*, {', '.join(params)}) -> MmaAtom: ..."
        )

    namespaces = []
    for namespace_name in ("sm80", "sm89", "sm90"):
        namespace = getattr(cuda, namespace_name)
        declarations = sorted(
            (name, value)
            for name, value in vars(type(namespace)).items()
            if isinstance(value, type) and issubclass(value, MmaAtom)
        )
        namespaces.append((namespace_name, namespace, declarations))

    lines = [
        "# Platform sub-namespaces (not OpSchema-backed).",
        "from tilefoundry_cuda.ir.tir.nn.mma_atom import MmaAtom as MmaAtom",
        "from tilefoundry_cuda.ir.tir.nn.wgmma import Form as Form, Major as Major",
        "from tilefoundry.ir.types import Mesh as Mesh",
        "",
    ]
    for namespace_name, namespace, declarations in namespaces:
        class_name = f"_Cuda{namespace_name.title()}"
        lines.append(f"class {class_name}:")
        enum_names = sorted(
            name
            for name, value in vars(type(namespace)).items()
            if isinstance(value, type) and issubclass(value, enum.Enum)
        )
        lines.extend(f"    {name}: type[{name}]" for name in enum_names)
        lines.extend(constructor(name, declaration) for name, declaration in declarations)
        lines.append("")
    lines.extend(
        [
            "class _Cuda:",
            "    sm80: _CudaSm80",
            "    sm89: _CudaSm89",
            "    sm90: _CudaSm90",
            "",
            "cuda: _Cuda",
        ]
    )
    return "\n".join(lines)


register_namespace("T", "cuda", cuda, stub=_stub)
