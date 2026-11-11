"""Record the three statements of the ISN version and prove each comparison fires.

Reads the pin from ``pyproject.toml``, the installed distribution version from
the running interpreter's metadata, and the active grammar snapshot from the
graph, then plants a mismatch in each comparison and records that it raised.
Run from the worktree root:

    PYTHONPATH=$PWD python docs/evidence/fragments/sn-pipeline-hygiene/sph-pin-test-run.py

Writes ``sph-pin-test.json`` beside this script.
"""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import grammar_sync
from tests.core.test_isn_pin_agreement import (
    ISN_DISTRIBUTION,
    installed_isn_version,
    pinned_isn_version,
    test_isn_pin_matches_the_installed_package,
)
from tests.graph.test_isn_grammar_snapshot_pin import (
    test_active_grammar_snapshot_matches_the_pin,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "sph-pin-test.json"


def _first_line(call: Callable[..., Any], *args: Any) -> str:
    """Run a comparison that must reject an injected value; return its message."""
    try:
        call(*args)
    except AssertionError as error:
        return str(error).splitlines()[0]
    raise SystemExit(f"comparison accepted the planted mismatch in {args!r}")


def _installed_mismatch(version: str) -> str:
    """Run the real pin/installed comparison with the metadata reporting ``version``."""
    real_version = importlib.metadata.version

    def planted(name: str, *args: Any, **kwargs: Any) -> str:
        return (
            version if name == ISN_DISTRIBUTION else real_version(name, *args, **kwargs)
        )

    importlib.metadata.version = planted
    try:
        return _first_line(test_isn_pin_matches_the_installed_package)
    finally:
        importlib.metadata.version = real_version


def main() -> int:
    pinned = pinned_isn_version()
    installed = installed_isn_version()

    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (v:ISNGrammarVersion {active: true}) "
            "RETURN v.version AS version, v.content_digest AS content_digest"
        )

    record = {
        "pinned": str(pinned),
        "installed": str(installed),
        "active_snapshot_versions": [row.get("version") for row in rows],
        "active_snapshot_digest": rows[0].get("content_digest") if rows else None,
        "snapshot_writer_module": grammar_sync.__name__,
        "active_snapshot_writer_file": grammar_sync.__file__,
        "planted_installed_mismatch": _installed_mismatch("0.11.2"),
        "planted_snapshot_mismatch": _first_line(
            test_active_grammar_snapshot_matches_the_pin, "0.11.0"
        ),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
