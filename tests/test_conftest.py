from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace


def test_pytest_configure_isolates_extensions_and_leaves_worker_locks(
    tmp_path, monkeypatch, pytestconfig
) -> None:
    conftest = next(
        plugin
        for plugin in pytestconfig.pluginmanager.get_plugins()
        if Path(getattr(plugin, "__file__", "")).resolve()
        == Path(__file__).with_name("conftest.py")
    )
    extensions = tmp_path / ".torch_extensions"
    lock = extensions / "residual_add" / "lock"
    lock.parent.mkdir(parents=True)
    lock.touch()

    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    conftest.pytest_configure(SimpleNamespace(rootpath=tmp_path))

    assert os.environ["TORCH_EXTENSIONS_DIR"] == str(extensions)
    assert not lock.exists()

    override = tmp_path / "operator_extensions"
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(override))
    conftest.pytest_configure(SimpleNamespace(rootpath=tmp_path))
    assert os.environ["TORCH_EXTENSIONS_DIR"] == str(override)

    lock.touch()
    conftest.pytest_configure(
        SimpleNamespace(rootpath=tmp_path, workerinput={})
    )
    assert lock.exists()
