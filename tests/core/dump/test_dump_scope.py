"""Scope / IDumper / ContextVar coverage."""

from __future__ import annotations

import asyncio
import threading

import pytest

from tilefoundry import dump


@pytest.mark.parametrize("content", ["kernel", lambda: "kernel"], ids=("str", "callable"))
def test_file_dumper_writes_files(tmp_path, content) -> None:
    """A real file lands at the nested path a subdir scope composes.

    A real file lands at the nested path a subdir scope composes, and a
    flag the parent masked off leaves nothing behind.

    Asserted against the filesystem rather than a MemoryDumper: the gate is what
    a developer inspecting a dump directory sees, and a masked write that still
    created the file would be invisible to an in-memory entry dict.
    """
    fd = dump.FileDumper(tmp_path / "scope")
    with dump.Scope(dumper=fd, flags=dump.Flags.CODEGEN_SOURCE):
        with dump.Scope("nested", dump.Flags.ALL):
            dump.dump("module.cu", content, dump.Flags.CODEGEN_SOURCE)
            dump.dump("ir.txt", lambda: pytest.fail("masked content was evaluated"), dump.Flags.PASS_IR)
    assert (tmp_path / "scope" / "nested" / "module.cu").read_text() == "kernel"
    assert not (tmp_path / "scope" / "nested" / "ir.txt").exists()


def test_dump_scope_isolates_tasks() -> None:
    """Child thread / asyncio.Task with its own ``Scope`` writes only into that scope's dumper.

    Child thread / asyncio.Task with its own ``Scope`` writes only
    into that scope's dumper; parent's ContextVar is untouched.
    """
    parent_dumper = dump.MemoryDumper()
    thread_results: list[dict[str, str | bytes]] = []

    def worker():
        local = dump.MemoryDumper()
        with dump.Scope(dumper=local, flags=dump.Flags.ALL):
            dump.dump("t.txt", "child", dump.Flags.PASS_IR)
        thread_results.append(dict(local.entries))

    with dump.Scope(dumper=parent_dumper, flags=dump.Flags.ALL):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        dump.dump("p.txt", "self", dump.Flags.PASS_IR)
    assert thread_results == [{"t.txt": "child"}]
    assert parent_dumper.entries == {"p.txt": "self"}

    async def task_local():
        local = dump.MemoryDumper()
        with dump.Scope(dumper=local, flags=dump.Flags.ALL):
            dump.dump("a.txt", "child", dump.Flags.PASS_IR)
        return local.entries

    parent2 = dump.MemoryDumper()

    async def driver():
        with dump.Scope(dumper=parent2, flags=dump.Flags.ALL):
            entries = await asyncio.create_task(task_local())
            dump.dump("p2.txt", "self", dump.Flags.PASS_IR)
        return entries

    entries = asyncio.run(driver())
    assert entries == {"a.txt": "child"}
    assert parent2.entries == {"p2.txt": "self"}
