"""The wiki discovery stage function and its thin click wrapper.

``run_wiki_stage`` carries the discovery body behind a frozen
:class:`WikiStageOptions`; the ``wiki`` click command builds those options and
calls the stage. These tests measure the settled surface:

- ``--scan-only`` selects the seeding half: the engine gets ``scan_only`` and
  page seeding runs.
- ``--flush`` selects the draining half: the engine gets ``score_only`` and page
  seeding is skipped.
- ``--topic`` is the free-text steer the old free-text ``--focus`` reached.
- ``--limit`` caps pages (the old ``--max-pages``).
- ``--wiki-site`` selects one configured site (the old ``--source``).
- ``--focus ITEMS`` is refused with a message stating the mechanism: the wiki
  claim query takes no item filter.

The engine is replaced at its single entry point
(``run_parallel_wiki_discovery``) so the kwargs it receives are the subject of
the assertion. ``run_discovery`` is replaced with one that drives ``async_main``
directly, keeping the measurement off the rich/plain harness and off the graph.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from imas_codex.cli.discover.wiki import WikiStageOptions, run_wiki_stage, wiki

FACILITY = "jt-60sa"

SITE_A = "https://wiki-a.example.org/"
SITE_B = "https://wiki-b.example.org/"

FACILITY_CONFIG = {
    "wiki_sites": [
        {"url": SITE_A, "name": "wiki-a", "site_type": "mediawiki"},
        {"url": SITE_B, "name": "wiki-b", "site_type": "mediawiki"},
    ],
}

_ENGINE_RESULT = {
    "scanned": 4,
    "scored": 0,
    "ingested": 0,
    "documents": 0,
    "images_scored": 0,
    "cost": 0.0,
    "elapsed_seconds": 0.5,
}


def _graph_client() -> MagicMock:
    """A GraphClient whose queries return no rows, so counts read as zero."""
    instance = MagicMock()
    instance.query.return_value = []
    context = MagicMock()
    context.__enter__.return_value = instance
    return MagicMock(return_value=context)


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the engine entry point and record the kwargs it receives."""
    calls: list[dict] = []
    bulk: list[dict] = []

    async def fake_engine(**kwargs):
        calls.append(kwargs)
        return dict(_ENGINE_RESULT)

    def fake_bulk_pages(**kwargs):
        bulk.append(kwargs)
        return 0

    monkeypatch.setattr(
        "imas_codex.discovery.wiki.parallel.run_parallel_wiki_discovery",
        fake_engine,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.wiki.parallel.bulk_discover_pages",
        fake_bulk_pages,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda facility: FACILITY_CONFIG,
    )
    # Zero pages in the graph means the seeding half runs, so the scan/flush
    # split is measurable rather than masked by a cached scan.
    monkeypatch.setattr(
        "imas_codex.discovery.wiki.get_wiki_stats",
        lambda facility: {"pages": 0},
    )
    monkeypatch.setattr(
        "imas_codex.discovery.wiki.graph_ops.create_doc_source",
        lambda gc, facility, **kwargs: "docsrc:1",
    )
    monkeypatch.setattr(
        "imas_codex.discovery.wiki.graph_ops.reset_transient_pages",
        lambda facility, **kwargs: {},
    )
    monkeypatch.setattr(
        "imas_codex.discovery.wiki.graph_ops.recover_failed_pages",
        lambda facility: 0,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.wiki.graph_ops.recover_failed_documents",
        lambda facility: 0,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.wiki.graph_ops.defer_failed_documents",
        lambda facility: 0,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.wiki.graph_ops.has_pending_work",
        lambda facility, **kwargs: True,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.wiki.graph_ops.has_pending_document_work",
        lambda facility, **kwargs: True,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.wiki.parallel.bulk_discover_documents",
        lambda **kwargs: (0, {}),
    )
    monkeypatch.setattr("imas_codex.graph.client.GraphClient", _graph_client())
    monkeypatch.setattr("imas_codex.graph.GraphClient", _graph_client())
    monkeypatch.setattr("imas_codex.cli.rich_output.should_use_rich", lambda: False)

    # Drive async_main directly, off the rich/plain harness.
    def fake_run_discovery(config, async_main, *, on_complete=None):
        result = asyncio.run(async_main(asyncio.Event(), None))
        if on_complete is not None:
            on_complete(result)
        return result

    monkeypatch.setattr(
        "imas_codex.cli.discover.common.run_discovery", fake_run_discovery
    )
    return {"calls": calls, "bulk": bulk}


def test_stage_options_are_frozen() -> None:
    options = WikiStageOptions()
    with pytest.raises(FrozenInstanceError):
        options.scan_only = True  # type: ignore[misc]


def test_scan_only_selects_the_seeding_half(engine) -> None:
    run_wiki_stage(FACILITY, WikiStageOptions(scan_only=True, wiki_site="0"))
    assert engine["bulk"], "the seeding half must enumerate pages"
    assert engine["calls"][-1]["scan_only"] is True
    assert engine["calls"][-1]["score_only"] is False


def test_flush_selects_the_draining_half(engine) -> None:
    run_wiki_stage(FACILITY, WikiStageOptions(flush=True, wiki_site="0"))
    assert engine["bulk"] == [], "the draining half must not enumerate pages"
    assert engine["calls"][-1]["score_only"] is True
    assert engine["calls"][-1]["scan_only"] is False


def test_topic_reaches_the_scorer(engine) -> None:
    run_wiki_stage(
        FACILITY,
        WikiStageOptions(scan_only=True, wiki_site="0", topic="equilibrium"),
    )
    assert engine["calls"][-1]["focus"] == "equilibrium"


def test_limit_caps_pages(engine) -> None:
    run_wiki_stage(FACILITY, WikiStageOptions(scan_only=True, wiki_site="0", limit=7))
    assert engine["calls"][-1]["page_limit"] == 7


def test_wiki_site_selects_one_site(engine) -> None:
    run_wiki_stage(FACILITY, WikiStageOptions(scan_only=True, wiki_site="1"))
    assert len(engine["calls"]) == 1
    assert engine["calls"][0]["base_url"] == SITE_B


def test_all_sites_run_without_wiki_site(engine) -> None:
    run_wiki_stage(FACILITY, WikiStageOptions(scan_only=True))
    assert [call["base_url"] for call in engine["calls"]] == [SITE_A, SITE_B]


def test_focus_items_are_refused_stating_the_mechanism(engine) -> None:
    with pytest.raises(click.UsageError) as excinfo:
        run_wiki_stage(FACILITY, WikiStageOptions(focus=("MAG/coil",)))
    message = str(excinfo.value)
    assert "claim query takes no item filter" in message
    assert "facility-discovery-sequence" not in message
    assert "section" not in message


def test_cli_focus_is_refused(engine) -> None:
    result = CliRunner().invoke(wiki, [FACILITY, "--focus", "MAG/coil"])
    assert result.exit_code != 0
    assert "claim query takes no item filter" in result.output
    assert "facility-discovery-sequence" not in result.output
    assert "section" not in result.output


def test_click_command_is_a_thin_wrapper() -> None:
    with patch("imas_codex.cli.discover.wiki.run_wiki_stage") as mock_stage:
        result = CliRunner().invoke(
            wiki,
            [
                FACILITY,
                "--scan-only",
                "--flush",
                "--topic",
                "eq",
                "--limit",
                "5",
                "-c",
                "2.5",
                "-s",
                "1",
                "--max-depth",
                "3",
                "--focus",
                "MAG/coil",
                "--rescan",
                "--rescan-documents",
                "--score-workers",
                "3",
                "--ingest-workers",
                "6",
                "--time",
                "7",
                "--store-images",
                "--min-score",
                "0.3",
                "--verbose",
            ],
        )
    assert result.exit_code == 0, result.output
    facility, options = mock_stage.call_args.args
    assert facility == FACILITY
    assert isinstance(options, WikiStageOptions)
    assert options.scan_only is True
    assert options.flush is True
    assert options.topic == "eq"
    assert options.limit == 5
    assert options.cost_limit == 2.5
    assert options.wiki_site == "1"
    assert options.max_depth == 3
    assert options.focus == ("MAG/coil",)
    assert options.rescan is True
    assert options.rescan_documents is True
    assert options.score_workers == 3
    assert options.ingest_workers == 6
    assert options.time_limit == 7
    assert options.store_images is True
    assert options.min_score == 0.3
    assert options.verbose is True


def test_score_only_is_a_deprecated_alias_for_flush() -> None:
    with patch("imas_codex.cli.discover.wiki.run_wiki_stage") as mock_stage:
        result = CliRunner().invoke(wiki, [FACILITY, "--score-only"])
    assert result.exit_code == 0, result.output
    options = mock_stage.call_args.args[1]
    assert options.flush is True


def test_cli_scan_only_reaches_the_engine(engine) -> None:
    result = CliRunner().invoke(wiki, [FACILITY, "--scan-only", "-s", "0"])
    assert result.exit_code == 0, result.output
    assert engine["calls"][-1]["scan_only"] is True
    assert engine["calls"][-1]["score_only"] is False


def test_cli_flush_reaches_the_engine(engine) -> None:
    result = CliRunner().invoke(wiki, [FACILITY, "--flush", "-s", "0"])
    assert result.exit_code == 0, result.output
    assert engine["calls"][-1]["score_only"] is True
    assert engine["calls"][-1]["scan_only"] is False
