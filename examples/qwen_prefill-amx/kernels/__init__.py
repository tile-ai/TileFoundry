"""Build the FFI entry against the supplied experimental llama.cpp runtime."""

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tvm_ffi
from tvm_ffi import libinfo

_module = None


def ops():
    global _module
    if _module is not None:
        return _module
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This runtime requires Apple Silicon macOS")
    source = os.environ.get("LLAMA_CPP")
    if not source:
        raise RuntimeError(
            "Set LLAMA_CPP to the supplied experimental llama.cpp checkout"
        )
    source = Path(source).expanduser().resolve()
    native = source / "ggml/src/ggml-blas"
    if not (native / "shared_fp16.hpp").is_file():
        raise RuntimeError(
            "LLAMA_CPP must contain the supplied AMX runtime (shared_fp16.hpp)"
        )
    root = Path(__file__).resolve().parent
    build = (
        Path(os.environ.get("LLAMA_BUILD", root / ".build/llama"))
        .expanduser()
        .resolve()
    )
    cmake = shutil.which("cmake") or str(Path(sys.executable).parent / "cmake")
    subprocess.run(
        [
            cmake,
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_METAL=OFF",
            "-DGGML_BLAS=ON",
            "-DGGML_BLAS_VENDOR=Apple",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [cmake, "--build", str(build), "--target", "llama", "ggml-blas", "-j", "8"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    library = build / "bin/libggml-blas.dylib"
    symbols = subprocess.check_output(["nm", "-gU", str(library)], text=True)
    if (
        "_tf_amx_load_prepared" not in symbols
        or "_tf_amx_clear_prepared" not in symbols
    ):
        raise RuntimeError(
            "The supplied runtime must export tf_amx_load_prepared and tf_amx_clear_prepared"
        )
    out = root / ".build/prefill.dylib"
    out.parent.mkdir(parents=True, exist_ok=True)
    ffi = Path(libinfo.find_libtvm_ffi())
    dependencies = (
        root / "prefill.cpp",
        ffi,
        library,
        build / "bin/libllama.dylib",
        *native.glob("*.hpp"),
    )
    if not out.exists() or any(
        p.stat().st_mtime > out.stat().st_mtime for p in dependencies
    ):
        clang = subprocess.check_output(
            ["xcrun", "--find", "clang++"], text=True
        ).strip()
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        subprocess.run(
            [
                clang,
                "-std=c++17",
                "-O3",
                "-shared",
                "-fPIC",
                "-isysroot",
                sdk,
                "-framework",
                "Accelerate",
                "-I" + str(Path(tvm_ffi.__file__).parent / "include"),
                "-I" + str(source / "include"),
                "-I" + str(source / "ggml/include"),
                "-I" + str(native),
                str(root / "prefill.cpp"),
                str(ffi),
                "-L" + str(build / "bin"),
                "-lllama",
                "-lggml-cpu",
                "-lggml-base",
                "-lggml",
                "-lggml-blas",
                "-Wl,-rpath," + str(build / "bin"),
                "-Wl,-rpath," + str(ffi.parent),
                "-o",
                str(out),
            ],
            check=True,
        )
    _module = tvm_ffi.load_module(str(out))
    return _module
