"""Native dependencies use the configured CMake search and fail when absent."""

import subprocess

import pytest

from tilefoundry.codegen import link_module
from tilefoundry_nvidia import cuda


@pytest.mark.parametrize("libdir", ("lib", "lib64"))
def test_nccl_search_uses_cmake_prefix_and_ignores_custom_environment(tmp_path, monkeypatch, libdir):
    prefix = tmp_path / "installation"
    (prefix / "include").mkdir(parents=True)
    (prefix / libdir).mkdir()
    (prefix / "include/nccl.h").touch()
    (prefix / libdir / "libnccl.so.2").touch()
    project = link_module.CMakeProject(tmp_path / "project", "lookup")
    cuda.codegen.NCCL().link(project)
    search = "\n".join(line for commands in project.appended for line in commands.splitlines()
                       if line.startswith(("find_path", "find_library")))
    script = tmp_path / "lookup.cmake"
    script.write_text('cmake_minimum_required(VERSION 3.18)\n' + search + '\n'
                      'file(WRITE "${OUTPUT}" "${NCCL_INCLUDE_DIR}\\n${NCCL_LIBRARY}\\n")\n')
    for name in ("NCCL_HOME", "NCCL_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / "invalid"))
    result = tmp_path / "found.txt"
    subprocess.run(["cmake", f"-DCMAKE_PREFIX_PATH={prefix}", f"-DOUTPUT={result}", "-P", str(script)], check=True)
    assert result.read_text().splitlines() == [str(prefix / "include"), str(prefix / libdir / "libnccl.so.2")]
    failed = subprocess.run(["cmake", "-DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=FALSE",
                             "-DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=FALSE",
                             "-DCMAKE_FIND_USE_CMAKE_SYSTEM_PATH=FALSE", "-P", str(script)],
                            capture_output=True, text=True)
    assert failed.returncode != 0
    assert "Could not find NCCL_INCLUDE_DIR" in failed.stderr
