"""A binding names the claimed source, and one refused binding fails alone.

The map stage echoes a source id from its own prompt text, which can carry a
mistyped facility prefix. A binding is persisted against the source its batch
was built for, never the model's text; a model entry naming a different source
is refused as an escalation naming both ids. A binding the graph still refuses
is recorded with its reason while its siblings, and the next IDS, persist, and
the run summary counts persisted bindings with a non-zero exit.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from imas_codex.ids import models, workers
from imas_codex.ids.mapping import validate_mappings
from imas_codex.ids.models import (
    SignalMappingBatch,
    SignalMappingEntry,
    TargetAssignment,
    TargetAssignmentBatch,
    ValidatedMappingResult,
    ValidatedSignalMapping,
    persist_mapping_result,
)

CLAIMED = "jt-60sa:eddbreadTime('E101173', 'DCSV', 'ppEFCC3FB', t1, t2)"
MISTYPED = "jt-60-sa:eddbreadTime('E101173', 'DCSV', 'ppEFCC3FB', t1, t2)"
SECTION = "pf_active/coil"
FIELD = "pf_active/coil/element/geometry/rectangle/r"


def _sections(source_id: str) -> TargetAssignmentBatch:
    return TargetAssignmentBatch(
        ids_name="pf_active",
        assignments=[
            TargetAssignment(
                source_id=source_id,
                imas_target_path=SECTION,
                confidence=0.9,
                reasoning="selected",
            )
        ],
    )


def _batch(
    entries: list[SignalMappingEntry], target_path: str = SECTION
) -> SignalMappingBatch:
    return SignalMappingBatch(
        ids_name="pf_active", target_path=target_path, mappings=entries
    )


def _entry(source_id: str, target_id: str) -> SignalMappingEntry:
    return SignalMappingEntry(
        source_id=source_id,
        target_id=target_id,
        transform_expression="value",
        confidence=0.9,
        reasoning="r",
    )


def _passing_report() -> MagicMock:
    report = MagicMock()
    report.escalations = []
    report.duplicate_targets = []
    report.all_passed = True
    report.binding_checks = []
    return report


def _validate(sections: TargetAssignmentBatch, batches: list[SignalMappingBatch]):
    with (
        patch(
            "imas_codex.ids.validation.validate_mapping",
            return_value=_passing_report(),
        ),
        patch("imas_codex.ids.validation.check_coverage_threshold", return_value=[]),
        patch("imas_codex.ids.tools.get_sign_flip_paths", return_value=set()),
    ):
        return validate_mappings(
            "jt-60sa", "pf_active", "4.1.1", sections, batches, gc=MagicMock()
        )


# ---------------------------------------------------------------------------
# 1. The binding names the claimed source, not the model's text
# ---------------------------------------------------------------------------


def test_mistyped_facility_prefix_persists_against_claimed_source():
    result = _validate(_sections(CLAIMED), [_batch([_entry(MISTYPED, FIELD)])])

    assert [b.source_id for b in result.bindings] == [CLAIMED]
    assert not any("refused" in e.reason for e in result.escalations)


def test_entry_naming_a_different_source_is_refused_naming_both_ids():
    other = "jt-60sa:aDifferentSignal"
    result = _validate(_sections(CLAIMED), [_batch([_entry(other, FIELD)])])

    assert result.bindings == []
    refused = [e for e in result.escalations if other in e.reason]
    assert len(refused) == 1
    assert CLAIMED in refused[0].reason


# ---------------------------------------------------------------------------
# 2. A refused binding fails alone
# ---------------------------------------------------------------------------


def _gc_writing_only(good_target: str) -> MagicMock:
    """A graph client whose MAPS_TO_IMAS MERGE reports a row only for a
    binding that matches a real target node."""
    gc = MagicMock()

    def _query(statement, **params):
        if "MERGE (sg)-[r:MAPS_TO_IMAS]->(ip)" in statement:
            return [{"written": 1}] if params.get("target_id") == good_target else []
        return []

    gc.query.side_effect = _query
    return gc


def test_refused_binding_recorded_while_sibling_persists():
    good = ValidatedSignalMapping(
        source_id="jt-60sa:a",
        target_id="pf_active/coil/current/data",
        confidence=0.9,
        mapping_type="direct",
    )
    absent = ValidatedSignalMapping(
        source_id="jt-60sa:a",
        target_id="pf_active/coil/missing",
        confidence=0.9,
        mapping_type="direct",
    )
    result = ValidatedMappingResult(
        facility="jt-60sa",
        ids_name="pf_active",
        dd_version="4.1.1",
        sections=[],
        bindings=[good, absent],
    )

    failures: list = []
    persist_mapping_result(
        result,
        gc=_gc_writing_only("pf_active/coil/current/data"),
        status="generated",
        binding_failures=failures,
    )

    assert [f.target_id for f in failures] == ["pf_active/coil/missing"]
    assert "missing" in failures[0].reason


def test_next_ids_persists_after_a_binding_failure(monkeypatch):
    """One refused binding does not sink its siblings or the next IDS."""
    sections_by_ids = {
        "pf_active": _sections("jt-60sa:pf"),
        "tf": _sections("jt-60sa:tf"),
    }
    bindings_by_ids = {
        "pf_active": [
            ValidatedSignalMapping(
                source_id="jt-60sa:pf",
                target_id="pf_active/coil/current/data",
                confidence=0.9,
                mapping_type="direct",
            ),
            ValidatedSignalMapping(
                source_id="jt-60sa:pf",
                target_id="pf_active/coil/missing",
                confidence=0.9,
                mapping_type="direct",
            ),
        ],
        "tf": [
            ValidatedSignalMapping(
                source_id="jt-60sa:tf",
                target_id="tf/coil/current/data",
                confidence=0.9,
                mapping_type="direct",
            )
        ],
    }

    state = workers.MappingDiscoveryState(
        facility="jt-60sa", target_ids_list=["pf_active", "tf"], skip_errors=True
    )
    for ids_name in ("pf_active", "tf"):
        sections = sections_by_ids[ids_name]
        batch = SignalMappingBatch(
            ids_name=ids_name,
            target_path=SECTION,
            mappings=[],
        )
        state.assignments[ids_name] = sections
        state.mapping_batches[ids_name] = [(sections.assignments[0], batch)]

    async def no_assembly(*_args, **_kwargs):
        return
        yield

    def fake_validate(_facility, ids_name, _dd, _sections, _batches, **_kwargs):
        return ValidatedMappingResult(
            facility="jt-60sa",
            ids_name=ids_name,
            dd_version="4.1.1",
            sections=_sections.assignments,
            bindings=bindings_by_ids[ids_name],
        )

    written: list[str] = []

    def fake_write(binding, gc):
        from imas_codex.ids.graph_ops import CandidateWriteError

        if binding.target_id == "pf_active/coil/missing":
            raise CandidateWriteError(
                f"wrote 0 MAPS_TO_IMAS edges for {binding.source_id} → "
                f"{binding.target_id}; the source or its IMASNode may not exist"
            )
        written.append(binding.target_id)
        return 1

    monkeypatch.setattr(workers, "GraphClient", lambda: MagicMock())
    monkeypatch.setattr("imas_codex.ids.mapping.adiscover_assembly", no_assembly)
    monkeypatch.setattr("imas_codex.ids.mapping.validate_mappings", fake_validate)
    monkeypatch.setattr(workers, "refresh_mapping_status", lambda *_args: "validated")
    monkeypatch.setattr(models, "write_mapping_binding", fake_write)

    asyncio.run(workers.validate_worker(state))

    assert state.ids_results["pf_active"] == {
        "bindings": 2,
        "persisted": 1,
        "failed": 1,
        "escalations": 0,
    }
    assert state.ids_results["tf"]["persisted"] == 1
    assert state.bindings_failed == 1
    assert state.bindings_persisted == 2
    assert "tf/coil/current/data" in written


# ---------------------------------------------------------------------------
# 3. The summary counts persisted bindings and the command exits non-zero
# ---------------------------------------------------------------------------


def test_run_summary_counts_persisted_and_exits_non_zero(monkeypatch):
    import imas_codex.cli.discover.common as common
    import imas_codex.cli.map as map_cli
    import imas_codex.graph.client as gclient
    import imas_codex.ids.tools as ids_tools
    import imas_codex.settings as settings

    printed: list[str] = []

    monkeypatch.setattr(
        ids_tools,
        "discover_mappable_ids",
        lambda *a, **k: {
            "available_domains": ["magnetic_field_systems"],
            "ids_targets": [{"ids_name": "pf_active"}],
            "total_sources": 1,
        },
    )
    monkeypatch.setattr(gclient, "GraphClient", lambda *a, **k: MagicMock())
    monkeypatch.setattr(settings, "get_dd_version", lambda: "4.1.1")
    monkeypatch.setattr(common, "use_rich_output", lambda: False)
    monkeypatch.setattr(common, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        common, "make_log_print", lambda domain, console=None: printed.append
    )
    monkeypatch.setattr(
        map_cli,
        "_run_plain_mode",
        lambda **kwargs: [
            {
                "ids_name": "pf_active",
                "bindings": 2,
                "persisted": 1,
                "failed": 1,
                "escalations": 0,
            }
        ],
    )

    result = CliRunner().invoke(map_cli.map_cmd, ["run", "jt-60sa", "-i", "pf_active"])

    assert result.exit_code == 1
    summary = "\n".join(str(line) for line in printed)
    assert "Bindings persisted: 1" in summary
    assert "Bindings failed: 1" in summary
