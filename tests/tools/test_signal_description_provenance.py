"""Name-only source text is not presented as a semantic description."""

from imas_codex.llm.search_formatters import format_signals_report
from imas_codex.llm.search_tools import _signal_rerank_state


def test_name_only_description_is_withheld_from_reranking_and_display():
    row = {
        "id": "demo:temperature",
        "name": "DEMO:TEMPERATURE",
        "accessor": "DEMO:TEMPERATURE",
        "description": "TEMPERATURE",
        "match_fields": ["name"],
        "description_source": "controlsmetadata:/variable_hmi",
        "description_source_updated_at": "2026-01-02 03:04:05",
    }
    assert _signal_rerank_state("temperature", row)["candidate"]["description"] == ""
    report = format_signals_report([row], [], {})
    assert "Matched: name" in report
    assert "Matched: description" not in report
    assert "\n  TEMPERATURE\n" not in report
    assert (
        "Metadata source: controlsmetadata:/variable_hmi; source updated: 2026-01-02 03:04:05"
        in report
    )
