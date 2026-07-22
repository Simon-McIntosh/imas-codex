"""Gap-only focus seeding and the ``--reseed`` opt-out.

``sn run --focus <file>`` defaults to *gap-only* seeding: a focused DD path
that already carries a live accepted/approved name is left untouched, so a
focused mop-up never churns names the catalog already holds. ``--reseed``
restores the reset-all behaviour (every focused path re-staged to ``pending``).

Section 1 — the partition helper (pure over a live graph view).
Section 2 — the CLI wiring: a manifest file expands to focus paths and the
gap-only / reseed switch drives whether accepted names are re-staged. The pool
orchestrator (``_run_sn_cmd``) is mocked so no LLM pipeline runs.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import partition_focus_by_accepted

PREFIX = "__focusreseedtest__"
ACCEPTED_PATH = f"{PREFIX}/accepted_leaf"
GAP_PATH = f"{PREFIX}/gap_leaf"
ACCEPTED_NAME = f"{PREFIX}_accepted_name"
GAP_NAME = f"{PREFIX}_gap_name"


def _cleanup() -> None:
    with GraphClient() as gc:
        gc.query(
            "MATCH (n) WHERE n.id CONTAINS $p DETACH DELETE n",
            p=PREFIX,
        )


@pytest.fixture
def focus_nodes():
    """Two DD paths: one with a live accepted name, one with a drafted name."""
    _cleanup()
    with GraphClient() as gc:
        gc.query(
            """
            MERGE (s1:StandardNameSource {id: $a_sns})
              SET s1.source_type='dd', s1.source_id=$a_path, s1.status='composed'
            MERGE (n1:StandardName {id: $a_name})
              SET n1.name_stage='accepted', n1.docs_stage='accepted'
            MERGE (s1)-[:PRODUCED_NAME]->(n1)
            MERGE (s2:StandardNameSource {id: $g_sns})
              SET s2.source_type='dd', s2.source_id=$g_path, s2.status='composed'
            MERGE (n2:StandardName {id: $g_name})
              SET n2.name_stage='drafted', n2.docs_stage='pending'
            MERGE (s2)-[:PRODUCED_NAME]->(n2)
            """,
            a_sns=f"dd:{ACCEPTED_PATH}",
            a_path=ACCEPTED_PATH,
            a_name=ACCEPTED_NAME,
            g_sns=f"dd:{GAP_PATH}",
            g_path=GAP_PATH,
            g_name=GAP_NAME,
        )
    yield
    _cleanup()


def _write_manifest(tmp_path):
    manifest = tmp_path / "focus.yaml"
    manifest.write_text(
        "schema_version: 1\n"
        "name: focus-reseed-test\n"
        "sources:\n"
        f"  {PREFIX}:\n"
        "    - accepted_leaf\n"
        "    - gap_leaf\n",
        encoding="utf-8",
    )
    return manifest


def _stub_seed(monkeypatch):
    """Bypass the real source-seed provenance guard.

    The seed step (``merge_standard_name_sources``) enforces an immutable
    provenance snapshot that synthetic test sources lack; it is orthogonal to
    the gap-only / reseed switch under test. The reset step (which flips
    ``name_stage``) runs via a direct query and stays real.
    """
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.merge_standard_name_sources",
        lambda sources, force=False: len(sources),
    )


def _name_stages() -> dict[str, str]:
    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (n:StandardName) WHERE n.id IN $ids "
            "RETURN n.id AS id, n.name_stage AS stage",
            ids=[ACCEPTED_NAME, GAP_NAME],
        )
    return {r["id"]: r["stage"] for r in rows}


# ── Section 1 — partition helper ──────────────────────────────────────────


def test_partition_empty_is_noop():
    # Empty input returns before touching the graph (safe in a default test).
    assert partition_focus_by_accepted([]) == ([], [])


@pytest.mark.graph
def test_partition_splits_accepted_from_gap(focus_nodes):
    gap, accepted = partition_focus_by_accepted([GAP_PATH, ACCEPTED_PATH])
    assert gap == [GAP_PATH]
    assert accepted == [ACCEPTED_PATH]


@pytest.mark.graph
def test_partition_preserves_input_order(focus_nodes):
    # A path with no source at all is a gap (never accepted).
    other = f"{PREFIX}/never_seen_leaf"
    gap, accepted = partition_focus_by_accepted([ACCEPTED_PATH, other, GAP_PATH])
    assert gap == [other, GAP_PATH]
    assert accepted == [ACCEPTED_PATH]


# ── Section 2 — CLI wiring (manifest file → focus → gap-only / reseed) ─────


@pytest.mark.graph
def test_focus_gap_only_leaves_accepted_untouched(focus_nodes, tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr("imas_codex.cli.sn._run_sn_cmd", lambda **kw: None)
    _stub_seed(monkeypatch)

    result = CliRunner().invoke(sn, ["run", "--focus", str(manifest), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Gap-only: skipping 1" in result.output
    stages = _name_stages()
    # Accepted name preserved; the gap (drafted) name re-staged to pending.
    assert stages[ACCEPTED_NAME] == "accepted"
    assert stages[GAP_NAME] == "pending"


@pytest.mark.graph
def test_focus_seeds_sources_with_exact_dd_version(focus_nodes, tmp_path, monkeypatch):
    """Every seeded focus source must carry an exact ``dd_version``.

    ``_pin_dd_source_snapshots`` refuses to infer ``latest`` for a genuinely
    new DD source, so the CLI must stamp the current DD version onto every
    source dict (mirroring the extraction worker). A missing stamp aborts the
    whole mop-up before any LLM spend the moment the manifest carries a source
    that was never seeded before.
    """
    from imas_codex.settings import get_dd_version

    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr("imas_codex.cli.sn._run_sn_cmd", lambda **kw: None)

    captured: list[dict] = []

    def _capture(sources, force=False):
        captured.extend(sources)
        return len(sources)

    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.merge_standard_name_sources",
        _capture,
    )

    result = CliRunner().invoke(sn, ["run", "--focus", str(manifest), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert captured, "no focus sources were seeded"
    current = get_dd_version()
    for src in captured:
        assert src.get("dd_version") == current, (
            f"seeded source {src.get('id')!r} lacks an exact dd_version"
        )


@pytest.mark.graph
def test_focus_reseed_restages_accepted(focus_nodes, tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr("imas_codex.cli.sn._run_sn_cmd", lambda **kw: None)
    _stub_seed(monkeypatch)

    result = CliRunner().invoke(
        sn, ["run", "--focus", str(manifest), "--reseed", "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert "Gap-only" not in result.output
    stages = _name_stages()
    # --reseed re-stages every focused path, accepted included.
    assert stages[ACCEPTED_NAME] == "pending"
    assert stages[GAP_NAME] == "pending"
