"""Build and cache the Apple Silicon prefill extension on first use."""

from pathlib import Path
import platform
import subprocess
import tvm_ffi
from tvm_ffi import libinfo
import sys
import tilefoundry

_module = None


def ops():
    global _module
    if _module is None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("The AMX example requires Apple Silicon macOS")
        root = Path(__file__).resolve().parent
        include = Path(sys.prefix) / "include"
        if not (include / "tilefoundry/runtime/amx/instructions.h").exists():
            include = Path(tilefoundry.__file__).resolve().parents[2] / "include"
        sources = (
            root / "prefill.cpp",
            root / "runtime.hpp",
            include / "tilefoundry/runtime/amx/instructions.h",
        )
        out = root / ".build/prefill.dylib"
        ffi = Path(libinfo.find_libtvm_ffi())
        if not out.exists() or any(
            p.stat().st_mtime > out.stat().st_mtime for p in sources
        ):
            out.parent.mkdir(exist_ok=True)
            clang = subprocess.check_output(
                ["xcrun", "--find", "clang++"], text=True
            ).strip()
            sdk = subprocess.check_output(
                ["xcrun", "--show-sdk-path"], text=True
            ).strip()
            subprocess.run(
                [
                    clang,
                    "-std=c++17",
                    "-O3",
                    "-shared",
                    "-fPIC",
                    "-isysroot",
                    sdk,
                    "-I" + str(Path(tvm_ffi.__file__).parent / "include"),
                    "-I" + str(include),
                    str(root / "prefill.cpp"),
                    str(ffi),
                    "-Wl,-rpath," + str(ffi.parent),
                    "-o",
                    str(out),
                ],
                check=True,
            )
        _module = tvm_ffi.load_module(str(out))
    return _module
