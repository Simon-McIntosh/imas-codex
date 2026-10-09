"""Signal claims stay within the requested signal and source identities.

``--focus`` names FacilitySignal identities (by id or accessor) and
SignalSource identities, the latter directly by id or through a member's
source array; a claim must then take only signals whose own identity or
source was named. These tests measure the two claim queries (enrichment and
check) and the stage that scopes them:

- the focus predicate is rendered into each claim query and its params;
- an empty focus renders no predicate, leaving the claim unfiltered;
- a source the run did not name stays unclaimed;
- naming one member of a grouped source claims every member of that source;
- the stage resolves focus items and refuses ones that name nothing here.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import click
import pytest

from imas_codex.cli.discover.signals import (
    SignalsStageOptions,
    _validate_focus,
    run_signals_stage,
)
from imas_codex.discovery.signals.parallel import (
    build_focus_predicate,
    claim_signals_for_check,
    claim_signals_for_enrichment,
)

SOURCE = "jt-60sa:coil-current-source"
SIGNAL = "jt-60sa:coil-current-signal"
ACCESSOR = "eddbreadTime('E101173', 'MMSYS', 'curCS1LKAT', t1, t2)"

CLAIMS = [claim_signals_for_enrichment, claim_signals_for_check]


class _Catalogue:
    """A toy signal catalogue that evaluates the focus filter of a claim.

    The claim query is the only place a signal's identity (id, accessor, or
    ``data_source_path`` array segment) and its source id are matched against
    ``$focus_items``. This fake resolves a focus item to the sources it names
    — directly by source id, or through any member's handle — and then claims
    every member of a named source, so a claim query with the focus filter
    dropped admits the whole catalogue, which is the defect the unnamed-source
    test exists to catch.
    """

    def __init__(self, signals: list[dict]) -> None:
        self.signals = signals
        self.selected: list[dict] = []

    def __enter__(self) -> _Catalogue:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def _named_sources(self, focus: set[str]) -> set[str]:
        arrays = {
            segment
            for signal in self.signals
            for segment in signal.get("data_source_path", "").split("/")
            if segment in focus
        }
        return {
            signal["source_id"]
            for signal in self.signals
            if signal["source_id"] in focus
            or any(
                segment in arrays
                for segment in signal.get("data_source_path", "").split("/")
            )
        }

    def query(self, cypher: str, **params: object) -> list[dict]:
        focus = params.get("focus_items")
        if "SET s.claimed_at" in cypher:
            if focus:
                assert "s.id IN $focus_items" in cypher, "focus filter dropped"
                item_set = set(focus)
                named = self._named_sources(item_set)
                self.selected = [
                    s
                    for s in self.signals
                    if s["id"] in item_set
                    or s["accessor"] in item_set
                    or s["source_id"] in named
                ]
            else:
                self.selected = list(self.signals)
            return []
        if "claim_token: $token" in cypher:
            return list(self.selected)
        return []


def test_focus_predicate_selects_signals_and_source_members() -> None:
    predicate = build_focus_predicate("s", [SOURCE, SIGNAL, ACCESSOR])
    assert "s.id IN $focus_items" in predicate
    assert "s.accessor IN $focus_items" in predicate
    assert "(s)-[:MEMBER_OF]->(source:SignalSource)" in predicate
    assert "source.id IN $focus_items" in predicate
    assert "split(coalesce(named.data_source_path, ''), '/')" in predicate


def test_empty_focus_renders_no_predicate() -> None:
    assert build_focus_predicate("s", []) == ""
    assert build_focus_predicate("s", None) == ""


@pytest.mark.parametrize("claim", CLAIMS)
def test_focus_reaches_each_claim_query(claim) -> None:
    focus = [SOURCE, SIGNAL, ACCESSOR]
    with patch("imas_codex.discovery.signals.parallel.GraphClient") as gc_class:
        graph = gc_class.return_value.__enter__.return_value
        graph.query.return_value = []
        claim("jt-60sa", focus_items=focus)

    claim_calls = [
        call
        for call in graph.query.call_args_list
        if "SET s.claimed_at" in call.args[0]
    ]
    assert len(claim_calls) == 1
    query = claim_calls[0].args[0]
    assert "s.id IN $focus_items" in query
    assert "s.accessor IN $focus_items" in query
    assert "(s)-[:MEMBER_OF]->(source:SignalSource)" in query
    assert "source.id IN $focus_items" in query
    assert "split(coalesce(named.data_source_path, ''), '/')" in query
    assert claim_calls[0].kwargs["focus_items"] == focus


@pytest.mark.parametrize("claim", CLAIMS)
def test_empty_focus_keeps_claim_unfiltered(claim) -> None:
    with patch("imas_codex.discovery.signals.parallel.GraphClient") as gc_class:
        graph = gc_class.return_value.__enter__.return_value
        graph.query.return_value = []
        claim("jt-60sa", focus_items=[])

    claim_calls = [
        call
        for call in graph.query.call_args_list
        if "SET s.claimed_at" in call.args[0]
    ]
    assert len(claim_calls) == 1
    assert "$focus_items" not in claim_calls[0].args[0]


@pytest.mark.parametrize("claim", CLAIMS)
def test_unnamed_source_in_the_same_facility_stays_unclaimed(claim) -> None:
    catalogue = _Catalogue(
        [
            {"id": "sig-named", "accessor": "A1", "source_id": "src-named"},
            {"id": "sig-unnamed", "accessor": "A2", "source_id": "src-unnamed"},
        ]
    )
    with patch(
        "imas_codex.discovery.signals.parallel.GraphClient", return_value=catalogue
    ):
        claim("jt-60sa", focus_items=["src-named"])

    assert [s["id"] for s in catalogue.selected] == ["sig-named"]


def test_named_source_claims_every_member() -> None:
    catalogue = _Catalogue(
        [
            {"id": "sig-a", "accessor": "A1", "source_id": "src-named"},
            {"id": "sig-b", "accessor": "A2", "source_id": "src-named"},
            {"id": "sig-c", "accessor": "A3", "source_id": "src-unnamed"},
        ]
    )
    with patch(
        "imas_codex.discovery.signals.parallel.GraphClient", return_value=catalogue
    ):
        claim_signals_for_enrichment("jt-60sa", focus_items=["src-named"])

    assert {s["id"] for s in catalogue.selected} == {"sig-a", "sig-b"}


def test_source_array_names_every_member_of_its_source() -> None:
    catalogue = _Catalogue(
        [
            {
                "id": "mdac10",
                "accessor": "A1",
                "source_id": "src-mdac",
                "data_source_path": "MDAC/magPbTC10",
            },
            {
                "id": "mdac11",
                "accessor": "A2",
                "source_id": "src-mdac",
                "data_source_path": "MDAC/magPbTC11",
            },
            {
                "id": "psrc10",
                "accessor": "A3",
                "source_id": "src-psrc",
                "data_source_path": "PSRC/magPbTC10",
            },
            {
                "id": "coil",
                "accessor": "A4",
                "source_id": "src-coil",
                "data_source_path": "MMSYS/curEF1LKAT",
            },
        ]
    )
    with patch(
        "imas_codex.discovery.signals.parallel.GraphClient", return_value=catalogue
    ):
        claim_signals_for_enrichment("jt-60sa", focus_items=["magPbTC10"])

    assert {s["id"] for s in catalogue.selected} == {"mdac10", "mdac11", "psrc10"}


def test_validate_focus_names_the_unknown_item() -> None:
    with patch("imas_codex.graph.GraphClient") as gc_class:
        gc_class.return_value.__enter__.return_value.query.return_value = [
            {"id": "known-source", "accessor": None, "data_source_path": None}
        ]
        with pytest.raises(click.UsageError) as excinfo:
            _validate_focus("jt-60sa", ["known-source", "not-a-source"])
    message = str(excinfo.value)
    assert "not-a-source" in message
    assert "known-source" not in message


def test_stage_refuses_an_unknown_focus_item() -> None:
    with patch("imas_codex.graph.GraphClient") as gc_class:
        gc_class.return_value.__enter__.return_value.query.return_value = []
        with pytest.raises(click.UsageError) as excinfo:
            run_signals_stage("jt-60sa", SignalsStageOptions(focus=("ghost",)))
    assert "ghost" in str(excinfo.value)


def test_stage_passes_focus_items_without_changing_topic(monkeypatch) -> None:
    captured = {}

    async def fake_engine(**kwargs):
        captured.update(kwargs)
        return {
            "scanned": 0,
            "enriched": 0,
            "checked": 0,
            "cost": 0.0,
            "elapsed_seconds": 0.0,
        }

    def fake_run_discovery(config, async_main, *, on_complete=None):
        return asyncio.run(async_main(asyncio.Event(), None))

    monkeypatch.setattr(
        "imas_codex.cli.discover.signals._validate_focus", lambda facility, items: None
    )
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda facility: {"ssh_host": "jt-60sa", "data_systems": {}},
    )
    monkeypatch.setattr(
        "imas_codex.discovery.signals.scanners.base.get_scanners_for_facility",
        lambda facility: [MagicMock(scanner_type="edas")],
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.ensure_remote_environment", lambda config: None
    )
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.run_discovery", fake_run_discovery
    )
    monkeypatch.setattr(
        "imas_codex.discovery.signals.parallel.run_parallel_data_discovery",
        fake_engine,
    )

    run_signals_stage(
        "jt-60sa", SignalsStageOptions(focus=(SOURCE,), topic="coil current")
    )
    assert captured["focus_items"] == [SOURCE]
    assert captured["focus"] == "coil current"
