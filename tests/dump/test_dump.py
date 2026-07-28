"""DumpScope / IDumper / ContextVar coverage."""
from __future__ import annotations

import asyncio
import threading

from tilefoundry.dump import (
    DumpFlags,
    DumpScope,
    FileDumper,
    MemoryDumper,
    dump,
)


def test_file_dumper_writes_files(tmp_path) -> None:
    """A real file lands at the nested path a subdir scope composes, and a
    flag the parent masked off leaves nothing behind.

    Asserted against the filesystem rather than a MemoryDumper: the gate is what
    a developer inspecting a dump directory sees, and a masked write that still
    created the file would be invisible to an in-memory entry dict.
    """
    fd = FileDumper(tmp_path / "scope")
    with DumpScope(dumper=fd, flags=DumpFlags.CODEGEN_SOURCE):
        with DumpScope("nested", DumpFlags.ALL):
            dump("module.cu", "kernel", DumpFlags.CODEGEN_SOURCE)
            dump("ir.txt", "pass", DumpFlags.PASS_IR)  # parent masked → drop
    assert (tmp_path / "scope" / "nested" / "module.cu").read_text() == "kernel"
    assert not (tmp_path / "scope" / "nested" / "ir.txt").exists()


def test_dump_scope_isolation_across_threads_and_asyncio_tasks() -> None:
    """Child thread / asyncio.Task with its own ``DumpScope`` writes only
    into that scope's dumper; parent's ContextVar is untouched."""
    parent_dumper = MemoryDumper()
    thread_results: list[dict[str, str | bytes]] = []

    def worker():
        local = MemoryDumper()
        with DumpScope(dumper=local, flags=DumpFlags.ALL):
            dump("t.txt", "child", DumpFlags.PASS_IR)
        thread_results.append(dict(local.entries))

    with DumpScope(dumper=parent_dumper, flags=DumpFlags.ALL):
        t = threading.Thread(target=worker)
        t.start(); t.join()  # noqa: E702
        dump("p.txt", "self", DumpFlags.PASS_IR)
    assert thread_results == [{"t.txt": "child"}]
    assert parent_dumper.entries == {"p.txt": "self"}

    async def task_local():
        local = MemoryDumper()
        with DumpScope(dumper=local, flags=DumpFlags.ALL):
            dump("a.txt", "child", DumpFlags.PASS_IR)
        return local.entries

    parent2 = MemoryDumper()

    async def driver():
        with DumpScope(dumper=parent2, flags=DumpFlags.ALL):
            entries = await asyncio.create_task(task_local())
            dump("p2.txt", "self", DumpFlags.PASS_IR)
        return entries

    entries = asyncio.run(driver())
    assert entries == {"a.txt": "child"}
    assert parent2.entries == {"p2.txt": "self"}
