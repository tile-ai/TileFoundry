"""Build the standalone hand-written AMX TVM FFI module."""

from pathlib import Path
import platform, subprocess
import tvm_ffi
from tvm_ffi import libinfo

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This AMX runtime requires Apple Silicon macOS")
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    clang = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    ffi = Path(libinfo.find_libtvm_ffi())
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
            str(ROOT / "native/prefill_ffi.cpp"),
            str(ffi),
            "-Wl,-rpath," + str(ffi.parent),
            "-o",
            str(out / "prefill.dylib"),
        ],
        check=True,
    )
    print(out / "prefill.dylib")
