"""The shared fixture's store teardown keeps its two promises.

Removing a Neo4j store costs one metadata operation per file, which is why the
fixture hands the recursive delete to a detached process instead of doing it in
the session's teardown. These tests pin the behaviour that buys: a removal that
outlasts the session's wait is detached rather than performed synchronously, and
a removal that fails is reported as a failure with its traceback kept, never as
a completed removal.

The fixture module is loaded here by path, under its own name, so the tests run
without a live Neo4j and without the fixture's own startup.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")

#: A remover that never touches the tree: it isolates the handoff from the
#: delete, so the test measures what the session does while a removal runs.
SLOW_REMOVER = "import time; time.sleep(1.5)"

#: A remover that fails after naming a reason, as a real one would.
FAILING_REMOVER = (
    "import sys; sys.stderr.write('cannot remove: read-only file\n'); sys.exit(3)"
)


def _load_fixture_module():
    name = "_standard_names_conftest_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CONFTEST)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _store(tmp_path: Path) -> tuple[tempfile.TemporaryDirectory, Path, Path]:
    directory = tempfile.TemporaryDirectory(prefix="disposable-neo4j-", dir=tmp_path)
    root = Path(directory.name)
    (root / "data").mkdir()
    (root / "data" / "neostore").write_text("store")
    return directory, root, root.with_name(f"{root.name}-removing")


def test_a_removal_that_outlasts_the_wait_is_detached(monkeypatch, tmp_path):
    module = _load_fixture_module()
    directory, root, trash = _store(tmp_path)
    remover_log = Path(f"{trash}.removal.log")
    monkeypatch.setattr(module, "_STORE_REMOVAL_BUDGET_SECONDS", 0.2)
    monkeypatch.setattr(module, "_STORE_REMOVER", SLOW_REMOVER)

    started = time.monotonic()
    report = module._detach_store_removal(directory)
    elapsed = time.monotonic() - started

    assert "handed to pid" in report, (
        f"a removal still running when the wait ended was not reported as "
        f"detached, so it was performed synchronously: {report}"
    )
    assert str(root) in report, report
    assert elapsed < 2.0, f"the session waited {elapsed:.2f}s for the delete"
    assert not root.exists(), "the store path survived the teardown"
    assert trash.exists(), (
        "a removal that outlasted the wait left no tree for the detached remover"
    )
    assert remover_log.exists(), "the detached remover's log was not left behind"


def test_the_session_teardown_uses_the_detached_removal():
    source = CONFTEST.read_text()
    fixture = source.split("def disposable_neo4j_uri", 1)[1]
    assert "_detach_store_removal(directory)" in fixture, (
        "the session fixture no longer hands the store removal off; a "
        "synchronous delete in teardown is what the removal budget protects"
    )
    assert "directory.cleanup()" not in fixture, (
        "the fixture's own teardown removes the store synchronously, which has "
        "no budget and defeats the handoff"
    )


def test_a_failed_removal_is_not_reported_as_a_removal(monkeypatch, tmp_path):
    module = _load_fixture_module()
    directory, root, trash = _store(tmp_path)
    remover_log = Path(f"{trash}.removal.log")
    monkeypatch.setattr(module, "_STORE_REMOVER", FAILING_REMOVER)

    with pytest.raises(RuntimeError) as raised:
        module._detach_store_removal(directory)

    assert str(remover_log) in str(raised.value), (
        f"the failure does not name the log holding the traceback: {raised.value}"
    )
    assert remover_log.exists(), "the failed remover's log was destroyed"
    assert "cannot remove" in remover_log.read_text(), (
        "the failed remover's output was not kept"
    )
    assert trash.exists(), "the unremoved tree was not left in place"


def test_a_completed_removal_is_reported_and_clears_its_log(monkeypatch, tmp_path):
    module = _load_fixture_module()
    directory, root, trash = _store(tmp_path)

    report = module._detach_store_removal(directory)

    assert "removed in" in report, report
    assert not root.exists(), "the store path survived the removal"
    assert not trash.exists(), "the trash tree survived a clean removal"
    assert not Path(f"{trash}.removal.log").exists(), (
        "a clean removal left its remover log behind"
    )
