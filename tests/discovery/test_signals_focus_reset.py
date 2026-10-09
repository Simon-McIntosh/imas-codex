"""A focused signals reset touches only the focused sources.

``discover signals --reset-to`` scopes its reset by scanner and category so a
scoped run never resets rows it will not then process. ``--focus`` must narrow
it the same way the enrichment and check claims are narrowed: naming a
``SignalSource`` resets every member of that source and leaves a signal of an
unnamed source at its current status.

The reset query is the only place the focus filter reaches the graph, so the
fake ``GraphClient`` below resolves the focus items to the sources they name
and then resets every member of a named source. A reset query with the focus
filter dropped admits the whole eligible cohort, which is the defect the
unnamed-source test exists to catch.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from imas_codex.cli.discover.signals import SignalsStageOptions, run_signals_stage

SOURCE = "jt-60sa:mdac-pb10-source"
UNNAMED_SOURCE = "jt-60sa:mdac-pb8-source"


class _Catalogue:
    """A toy signal catalogue that evaluates the focus filter of the reset.

    The reset query matches a signal's own identity (id or accessor) or the id
    of its ``SignalSource`` — directly, or through a member's source array —
    against ``$focus_items``. This fake mirrors that: a focus item resolves to
    the sources it names, and every member of a named source is reset. With no
    focus, every eligible row is reset.
    """

    def __init__(self, signals: list[dict]) -> None:
        self.signals = signals
        self.query_text = ""
        self.params: dict = {}
        self.reset_ids: list[str] = []

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
        if "RETURN count(n) AS reset_count" not in cypher:
            return []
        self.query_text = cypher
        self.params = params
        focus = params.get("focus_items")
        eligible = [
            signal
            for signal in self.signals
            if signal["status"] in params["source_statuses"]
        ]
        if focus:
            assert "n.id IN $focus_items" in cypher, "focus filter dropped"
            items = set(focus)
            named = self._named_sources(items)
            self.reset_ids = [
                signal["id"]
                for signal in eligible
                if signal["id"] in items
                or signal["accessor"] in items
                or signal["source_id"] in named
            ]
        else:
            self.reset_ids = [signal["id"] for signal in eligible]
        for signal in self.signals:
            if signal["id"] in self.reset_ids:
                signal["status"] = params["target_status"]
        return [{"reset_count": len(self.reset_ids)}]


def _install_stage_environment(monkeypatch, catalogue: _Catalogue) -> None:
    async def fake_engine(**kwargs: object) -> dict:
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
        "imas_codex.cli.discover.signals._validate_focus",
        lambda facility, items: None,
        raising=True,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda facility: {"ssh_host": "jt-60sa", "data_systems": {}},
        raising=True,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.signals.scanners.base.get_scanners_for_facility",
        lambda facility: [MagicMock(scanner_type="edas")],
        raising=True,
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.ensure_remote_environment",
        lambda config: None,
        raising=True,
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.use_rich_output", lambda: False, raising=True
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.run_discovery",
        fake_run_discovery,
        raising=True,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.signals.parallel.run_parallel_data_discovery",
        fake_engine,
        raising=True,
    )
    monkeypatch.setattr(
        "imas_codex.graph.GraphClient", lambda *a, **k: catalogue, raising=True
    )


def _run_reset(monkeypatch, signals: list[dict], focus: tuple[str, ...]) -> _Catalogue:
    catalogue = _Catalogue(signals)
    _install_stage_environment(monkeypatch, catalogue)
    run_signals_stage(
        "jt-60sa",
        SignalsStageOptions(focus=focus, reset_to="discovered"),
    )
    return catalogue


def test_focused_reset_passes_focus_filter_into_reset(monkeypatch) -> None:
    signals = [
        {"id": "sig-pb10", "accessor": "A1", "source_id": SOURCE, "status": "enriched"},
        {
            "id": "sig-pb8",
            "accessor": "A2",
            "source_id": UNNAMED_SOURCE,
            "status": "enriched",
        },
    ]
    catalogue = _run_reset(monkeypatch, signals, (SOURCE,))

    assert "n.id IN $focus_items" in catalogue.query_text
    assert "n.accessor IN $focus_items" in catalogue.query_text
    assert "(n)-[:MEMBER_OF]->(source:SignalSource)" in catalogue.query_text
    assert catalogue.params["focus_items"] == [SOURCE]


def test_unnamed_source_signal_keeps_its_status(monkeypatch) -> None:
    signals = [
        {"id": "sig-pb10", "accessor": "A1", "source_id": SOURCE, "status": "enriched"},
        {
            "id": "sig-pb8",
            "accessor": "A2",
            "source_id": UNNAMED_SOURCE,
            "status": "enriched",
        },
    ]
    catalogue = _run_reset(monkeypatch, signals, (SOURCE,))

    assert catalogue.reset_ids == ["sig-pb10"]
    by_id = {signal["id"]: signal for signal in signals}
    assert by_id["sig-pb10"]["status"] == "discovered"
    assert by_id["sig-pb8"]["status"] == "enriched"


def test_reset_without_focus_leaves_no_focus_filter(monkeypatch) -> None:
    signals = [
        {"id": "sig-pb10", "accessor": "A1", "source_id": SOURCE, "status": "enriched"},
        {
            "id": "sig-pb8",
            "accessor": "A2",
            "source_id": UNNAMED_SOURCE,
            "status": "enriched",
        },
    ]
    catalogue = _run_reset(monkeypatch, signals, ())

    assert "$focus_items" not in catalogue.query_text
    assert set(catalogue.reset_ids) == {"sig-pb10", "sig-pb8"}
