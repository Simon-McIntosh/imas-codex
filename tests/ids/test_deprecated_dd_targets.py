"""A DD path retired at or before the mapping's version is never offered.

The graph keeps a node for every DD version it ever held, so a node's existence
does not mean the configured Data Dictionary contains it.
``tf/coil/conductor/current/data`` was introduced in 3.22.0 and deprecated in
3.42.0, so it is not part of a 4.1.1 mapping, while ``tf/coil/current/data`` is.
One lifecycle predicate
(:func:`imas_codex.ids.graph_ops.dd_path_live_at`) is applied at three points —
candidate retrieval, binding validation and the hand-off export — and the
retired path must be excluded, refused and written unexpanded at each.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imas_codex.ids.graph_ops import dd_path_live_at
from imas_codex.ids.mapping import validate_mappings
from imas_codex.ids.models import (
    EscalationSeverity,
    SignalMappingBatch,
    SignalMappingEntry,
    TargetAssignment,
    TargetAssignmentBatch,
)
from imas_codex.models.constants import SearchMode
from imas_codex.search.search_strategy import SearchHit

DD_VERSION = "4.1.1"
CONDUCTOR = "tf/coil/conductor/current/data"
COIL = "tf/coil/current/data"
SECTION = "tf/coil"

# Lifecycle as the graph holds it. The conductor path was retired in a DD the
# mapping does not target; the coil path is still live at 4.1.1.
_LIFECYCLE = {
    COIL: {"introduced": "3.22.0", "deprecated": None},
    CONDUCTOR: {"introduced": "3.22.0", "deprecated": "3.42.0"},
}


def _lifecycle_rows(paths):
    # ``id`` keys the standalone lifecycle read; ``path`` keys the structure
    # read that carries the same columns alongside the target's data type.
    return [
        {
            "id": path,
            "path": path,
            "introduced": _LIFECYCLE[path]["introduced"],
            "deprecated": _LIFECYCLE[path]["deprecated"],
        }
        for path in paths
        if path in _LIFECYCLE
    ]


def _gc() -> MagicMock:
    """Fake graph answering only the lifecycle and parent-doc reads."""
    gc = MagicMock()

    def _query(statement: str, **params):
        paths = params.get("paths") or []
        if "INTRODUCED_IN" in statement and "DEPRECATED_IN" in statement:
            return _lifecycle_rows(paths)
        if "parent_documentation" in statement:
            return [{"id": path, "parent_documentation": ""} for path in paths]
        return []

    gc.query.side_effect = _query
    return gc


# ---------------------------------------------------------------------------
# The predicate itself
# ---------------------------------------------------------------------------


def test_predicate_rejects_a_path_deprecated_at_or_before_the_version():
    assert dd_path_live_at("3.22.0", "3.42.0", DD_VERSION) is False


def test_predicate_accepts_a_path_with_no_deprecation():
    assert dd_path_live_at("3.22.0", None, DD_VERSION) is True


def test_predicate_rejects_a_path_introduced_after_the_version():
    assert dd_path_live_at("4.2.0", None, DD_VERSION) is False


def test_predicate_is_disabled_without_a_version():
    assert dd_path_live_at("3.22.0", "3.42.0", None) is True


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------


def _hit(path: str, score: float) -> SearchHit:
    return SearchHit(
        path=path,
        ids_name=path.split("/", 1)[0],
        documentation=f"documentation for {path}",
        score=score,
        rank=1,
        search_mode=SearchMode.AUTO,
    )


class _FakeEncoder:
    def __init__(self) -> None:
        pass

    def embed_texts(self, texts, *, prompt_name=None, **kwargs):
        return np.zeros((len(texts), 4))


def _candidates(monkeypatch):
    from imas_codex.ids.candidates import retrieve_candidates

    monkeypatch.setattr(
        "imas_codex.embeddings.encoder.Encoder", _FakeEncoder, raising=True
    )

    def fake_hybrid(gc, query, *, ids_filter=None, dd_version=None, k=20, **kwargs):
        return [_hit(CONDUCTOR, 0.9), _hit(COIL, 0.8)]

    monkeypatch.setattr(
        "imas_codex.ids.candidates.hybrid_dd_search", fake_hybrid, raising=True
    )
    return retrieve_candidates(
        {"src-1": "toroidal field coil current"},
        {"src-1": ["tf"]},
        gc=_gc(),
        dd_version=DD_VERSION,
    )


def test_retired_path_is_excluded_from_candidates(monkeypatch):
    result = _candidates(monkeypatch)
    paths = [cand.hit.path for cand in result["src-1"]]
    assert CONDUCTOR not in paths


def test_live_path_passes_the_candidate_filter(monkeypatch):
    result = _candidates(monkeypatch)
    assert COIL in [cand.hit.path for cand in result["src-1"]]


# ---------------------------------------------------------------------------
# Binding validation
# ---------------------------------------------------------------------------

SOURCE = "jt-60sa:tf-current"


def _entry(target_id: str) -> SignalMappingEntry:
    return SignalMappingEntry(
        source_id=SOURCE,
        source_property="value",
        target_id=target_id,
        transform_expression="value",
        confidence=0.9,
        reasoning="r",
    )


def _batch(target_id: str) -> SignalMappingBatch:
    return SignalMappingBatch(
        ids_name="tf", target_path=SECTION, mappings=[_entry(target_id)]
    )


def _sections() -> TargetAssignmentBatch:
    return TargetAssignmentBatch(
        ids_name="tf",
        assignments=[
            TargetAssignment(
                source_id=SOURCE,
                imas_target_path=SECTION,
                confidence=0.9,
                reasoning="selected",
            )
        ],
    )


def _validate(target_id: str):
    passing = MagicMock()
    passing.escalations = []
    passing.duplicate_targets = []
    passing.all_passed = True
    passing.binding_checks = []
    with (
        patch("imas_codex.ids.validation.validate_mapping", return_value=passing),
        patch("imas_codex.ids.validation.check_coverage_threshold", return_value=[]),
        patch("imas_codex.ids.tools.get_sign_flip_paths", return_value=set()),
    ):
        return validate_mappings(
            "jt-60sa", "tf", DD_VERSION, _sections(), [_batch(target_id)], gc=_gc()
        )


def test_retired_path_is_refused_in_validation_naming_its_deprecation_version():
    result = _validate(CONDUCTOR)

    assert result.bindings == []
    refused = [e for e in result.escalations if e.severity == EscalationSeverity.ERROR]
    assert len(refused) == 1
    assert refused[0].target_id == CONDUCTOR
    assert "3.42.0" in refused[0].reason


def test_live_path_passes_validation():
    result = _validate(COIL)

    assert [b.target_id for b in result.bindings] == [COIL]
    refused = [e for e in result.escalations if e.severity == EscalationSeverity.ERROR]
    assert refused == []


# ---------------------------------------------------------------------------
# Hand-off export
# ---------------------------------------------------------------------------


def _build_handoff():
    from imas_codex.ids.handoff import build_mapping_handoff

    mapping = {
        "mapping": {
            "id": "jt-60sa:tf",
            "facility_id": "jt-60sa",
            "ids_name": "tf",
            "dd_version": DD_VERSION,
            "status": "generated",
            "provider": "imas-codex",
        },
        "bindings": [
            {"source_id": "jt-60sa:tf:conductor", "target_id": CONDUCTOR},
            {"source_id": "jt-60sa:tf:coil", "target_id": COIL},
        ],
    }

    gc = MagicMock()

    def _query(statement: str, **params):
        if "MEMBER_OF" in statement:
            # The member read carries each binding's target lifecycle, so the
            # retired target's row is present even with no FacilitySignal.
            return [
                {
                    "source_id": "jt-60sa:tf:conductor",
                    "target_id": CONDUCTOR,
                    "signal_id": None,
                    "data_source": None,
                    "data_source_path": None,
                    "source_property": "value",
                    "mapping_type": "direct",
                    "derived_from": None,
                    "confidence": None,
                    "evidence": None,
                    "introduced": _LIFECYCLE[CONDUCTOR]["introduced"],
                    "deprecated": _LIFECYCLE[CONDUCTOR]["deprecated"],
                },
                {
                    "source_id": "jt-60sa:tf:coil",
                    "target_id": COIL,
                    "signal_id": "jt-60sa:tf:coil:1",
                    "data_source": "edas",
                    "data_source_path": "TF/coil1",
                    "source_property": "value",
                    "mapping_type": "direct",
                    "derived_from": None,
                    "confidence": 0.9,
                    "evidence": "",
                    "introduced": _LIFECYCLE[COIL]["introduced"],
                    "deprecated": _LIFECYCLE[COIL]["deprecated"],
                },
            ]
        return []

    gc.query.side_effect = _query
    with patch("imas_codex.ids.handoff.search_existing_mappings", return_value=mapping):
        return build_mapping_handoff("jt-60sa", ["tf"], gc=gc)


def test_retired_path_is_written_unexpanded_in_the_export():
    document = _build_handoff()
    entry = document["ids"][0]

    unexpanded = {row["target_path"]: row for row in entry["unexpanded"]}
    assert CONDUCTOR in unexpanded
    assert "3.42.0" in unexpanded[CONDUCTOR]["reason"]
    assert all(sig["target_path"] != CONDUCTOR for sig in entry["signals"])


def test_live_path_is_expanded_in_the_export():
    document = _build_handoff()
    entry = document["ids"][0]

    assert [sig["target_path"] for sig in entry["signals"]] == [COIL]
    assert all(row["target_path"] != COIL for row in entry["unexpanded"])


# ---------------------------------------------------------------------------
# Live graph confirmation
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_lifecycle_predicate_on_the_live_graph():
    from imas_codex.graph.client import GraphClient
    from imas_codex.ids.graph_ops import dd_path_lifecycles

    with GraphClient() as gc:
        lifecycles = dd_path_lifecycles(gc, [CONDUCTOR, COIL], DD_VERSION)

    assert lifecycles[CONDUCTOR].live is False
    assert lifecycles[CONDUCTOR].deprecated_version == "3.42.0"
    assert lifecycles[COIL].live is True
