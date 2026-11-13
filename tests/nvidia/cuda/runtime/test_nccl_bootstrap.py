"""A generated program owns its NCCL dependency and communicator bootstrap ABI."""

import subprocess

import pytest

from tilefoundry_nvidia import cuda


def test_bootstrap_uses_the_supplied_library_despite_invalid_installation_hints(tmp_path, monkeypatch):
    source = tmp_path / "bootstrap.c"
    library = tmp_path / "program.so"
    source.write_text('''
#include <stdint.h>
#include <string.h>
typedef struct { char internal[128]; } UniqueId;
const char *ncclGetErrorString(int status) { return "bootstrap failure"; }
int ncclGetUniqueId(UniqueId *id) { memset(id, 42, sizeof(*id)); return 0; }
int ncclCommInitRank(void **comm, int size, UniqueId id, int rank) {
    if (id.internal[0] != 42) return 1;
    *comm = (void *)(uintptr_t)(size * 10 + rank);
    return 0;
}
int ncclCommDestroy(void *comm) { return (uintptr_t)comm == 21 ? 0 : 1; }
''')
    subprocess.run(["gcc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    monkeypatch.setenv("NCCL_HOME", str(tmp_path / "missing-home"))
    monkeypatch.setenv("NCCL_ROOT", str(tmp_path / "missing-root"))
    linked = cuda.nccl.Library(library)
    token = linked.unique_id()
    communicator = linked.create(2, token, 1)
    assert communicator == 21
    linked.destroy(communicator)
    with pytest.raises(ValueError, match="valid local rank"):
        linked.create(2, token, 2)
    with pytest.raises(ValueError, match="unique ID"):
        linked.create(2, token[:-1], 1)
    with pytest.raises(RuntimeError, match="bootstrap failure"):
        linked.destroy(12)
    with pytest.raises(FileNotFoundError):
        cuda.nccl.Library(tmp_path / "absent.so")


def test_bootstrap_rejects_a_program_without_nccl(tmp_path):
    source = tmp_path / "ordinary.c"
    library = tmp_path / "ordinary.so"
    source.write_text("int ordinary(void) { return 0; }\n")
    subprocess.run(["gcc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    with pytest.raises(RuntimeError, match="no linked NCCL bootstrap symbols"):
        cuda.nccl.Library(library)
