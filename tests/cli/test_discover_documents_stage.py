"""Drive the documents discovery stage and its thin click wrapper.

The stage splits the documents pipeline into a seeding half (scan scored
facility paths and create ``Document`` nodes) and a draining half (fetch the
image Documents and run VLM captioning and scoring over them). ``--scan-only``
runs the seeding half and stops; ``--flush`` runs the draining half without
seeding. The click command is driven through the real ``discover`` group, so a
wrapper that stopped calling the stage function would fail here.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib

import pytest
from click.testing import CliRunner

from imas_codex.cli.discover import discover
from imas_codex.cli.discover.documents import DocumentsOptions, run_documents_stage

# ``imas_codex.cli.discover.documents`` is shadowed in the package namespace by
# the click Command of the same name, so patch the module through sys.modules.
_DOCUMENTS_MODULE = importlib.import_module("imas_codex.cli.discover.documents")

FACILITY = "tcv"
SSH_HOST = "tcv.example.org"


class _Recorder:
    """Captures what each half of the stage was asked to do."""

    def __init__(self) -> None:
        self.scan_calls: list[dict] = []
        self.pipeline_states: list = []


@pytest.fixture
def documents_env(monkeypatch):
    """Replace the scanner, the engine and facility config with recorders."""
    rec = _Recorder()
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda name: {"id": name, "ssh_host": SSH_HOST},
    )

    def fake_scan(
        facility,
        min_score=0.5,
        max_paths=100,
        ssh_host=None,
        progress_callback=None,
    ):
        rec.scan_calls.append(
            {
                "facility": facility,
                "min_score": min_score,
                "max_paths": max_paths,
                "ssh_host": ssh_host,
            }
        )
        return {
            "total_files": 3,
            "total_paths": 2,
            "new_files": 3,
            "skipped_files": 0,
        }

    monkeypatch.setattr(
        "imas_codex.discovery.documents.scanner.scan_facility_documents",
        fake_scan,
    )

    async def fake_pipeline(
        state,
        *,
        num_image_workers=2,
        num_vlm_workers=1,
        stop_event=None,
        on_worker_status=None,
    ):
        rec.pipeline_states.append(state)
        return {
            "images_fetched": 0,
            "images_captioned": 0,
            "cost": 0.0,
            "elapsed_seconds": 0.0,
        }

    monkeypatch.setattr(
        "imas_codex.discovery.documents.pipeline.run_document_discovery",
        fake_pipeline,
    )

    def fake_run_discovery(config, async_main, *, on_complete=None):
        asyncio.run(async_main(None, None))
        return {}

    monkeypatch.setattr(
        "imas_codex.cli.discover.common.run_discovery", fake_run_discovery
    )
    return rec


def test_options_dataclass_is_frozen() -> None:
    options = DocumentsOptions()
    assert dataclasses.is_dataclass(options)
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.limit = 1  # type: ignore[misc]


def test_command_is_a_thin_wrapper_that_builds_the_options(monkeypatch) -> None:
    captured: dict = {}

    def fake_stage(facility, options):
        captured["facility"] = facility
        captured["options"] = options

    monkeypatch.setattr(_DOCUMENTS_MODULE, "run_documents_stage", fake_stage)

    result = CliRunner().invoke(
        discover,
        [
            "documents",
            FACILITY,
            "--limit",
            "7",
            "-c",
            "1.5",
            "--topic",
            "equilibrium",
            "--scan-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["facility"] == FACILITY
    options = captured["options"]
    assert isinstance(options, DocumentsOptions)
    assert options.limit == 7
    assert options.cost_limit == 1.5
    assert options.topic == "equilibrium"
    assert options.scan_only is True
    assert options.flush is False


def test_scan_only_runs_one_seeding_half_via_the_command(documents_env) -> None:
    result = CliRunner().invoke(discover, ["documents", FACILITY, "--scan-only"])

    assert result.exit_code == 0, result.output
    assert [call["facility"] for call in documents_env.scan_calls] == [FACILITY]
    assert documents_env.pipeline_states == []


def test_flush_runs_only_the_draining_half_via_the_command(documents_env) -> None:
    result = CliRunner().invoke(discover, ["documents", FACILITY, "--flush"])

    assert result.exit_code == 0, result.output
    assert documents_env.scan_calls == []
    assert len(documents_env.pipeline_states) == 1


def test_stage_flush_skips_the_seeding_half(documents_env) -> None:
    run_documents_stage(FACILITY, DocumentsOptions(flush=True))

    assert documents_env.scan_calls == []
    assert len(documents_env.pipeline_states) == 1


def test_stage_scan_only_stops_after_the_seeding_half(documents_env) -> None:
    run_documents_stage(FACILITY, DocumentsOptions(scan_only=True))

    assert len(documents_env.scan_calls) == 1
    assert documents_env.pipeline_states == []


def test_topic_reaches_the_scorer(documents_env) -> None:
    result = CliRunner().invoke(
        discover, ["documents", FACILITY, "--topic", "diagnostics"]
    )

    assert result.exit_code == 0, result.output
    assert len(documents_env.pipeline_states) == 1
    assert documents_env.pipeline_states[0].focus == "diagnostics"


def test_limit_caps_items(documents_env) -> None:
    result = CliRunner().invoke(discover, ["documents", FACILITY, "--limit", "3"])

    assert result.exit_code == 0, result.output
    assert documents_env.scan_calls[0]["max_paths"] == 3


def test_focus_is_refused_naming_the_plan_section(documents_env) -> None:
    result = CliRunner().invoke(
        discover, ["documents", FACILITY, "--focus", "equilibrium"]
    )

    assert result.exit_code != 0
    assert "facility-discovery-sequence" in result.output
    assert "§7" in result.output


def test_scan_only_and_flush_together_are_refused(documents_env) -> None:
    result = CliRunner().invoke(
        discover, ["documents", FACILITY, "--scan-only", "--flush"]
    )

    assert result.exit_code != 0
    assert documents_env.scan_calls == []
    assert documents_env.pipeline_states == []


def test_command_help_lists_settled_spellings() -> None:
    result = CliRunner().invoke(discover, ["documents", "--help"])

    assert result.exit_code == 0, result.output
    for flag in ("--scan-only", "--flush", "--topic", "--limit"):
        assert flag in result.output
    # The retired per-domain item cap is replaced by --limit.
    assert "--max-paths" not in result.output
