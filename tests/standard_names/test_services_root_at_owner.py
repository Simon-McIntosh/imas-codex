"""A storage-layer reader installs the graph services that it reads.

The storage modules reach their collaborators through a composition root. A
reader that depends on some other module having imported the graph facade
first refuses whenever an entrypoint — a CLI command, a worker task, a test —
imports a storage module directly, and the refusal reads like a credential or
availability fault rather than a wiring one. The probe below runs in a fresh
interpreter so that no other test in the session can have installed the root
beforehand, and calls a reader with a client that is never meant to answer.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_READER_PROBE = textwrap.dedent(
    """
    import sys

    from imas_codex.standard_names.graph_store.graph_supersession import (
        classify_sourceless_standard_names,
    )

    if "imas_codex.standard_names.graph_ops" in sys.modules:
        print("the facade was imported before the reader; this probe cannot measure")
        raise SystemExit(4)


    class UnqueriedClient:
        def query(self, *args, **kwargs):
            raise AssertionError("the reader reached the client")


    try:
        classify_sourceless_standard_names(UnqueriedClient())
    except AssertionError:
        print("the reader installed the services it reads")
        raise SystemExit(0)
    except RuntimeError as exc:
        print(f"the reader refused: {exc}")
        raise SystemExit(3)
    print("the reader neither read the services nor reached the client")
    raise SystemExit(5)
    """
)


# The cost of this case is the fresh interpreter and the package import, not the
# assertion: about 17 s alone on an idle node. A contended node stretches that
# by however much the run is squeezed, so the project's 30 s default sits inside
# the ordinary spread and the ceiling is raised for the same reason the other
# fresh-process import contract raises it.
@pytest.mark.timeout(300)
def test_a_storage_reader_installs_the_services_it_reads() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _READER_PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"exit {result.returncode}: {result.stdout.strip()} {result.stderr.strip()}"
    )
