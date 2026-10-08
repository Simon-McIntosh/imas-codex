"""Wiki ingestion threshold settings."""

from imas_codex import settings


def test_wiki_ingest_threshold_defaults_to_point_sixteen(monkeypatch) -> None:
    monkeypatch.delenv("IMAS_CODEX_WIKI_INGEST_THRESHOLD", raising=False)
    monkeypatch.setattr(settings, "_get_section", lambda section: {})
    assert settings.get_wiki_ingest_threshold() == 0.16


def test_wiki_ingest_threshold_reads_discovery_setting(monkeypatch) -> None:
    monkeypatch.delenv("IMAS_CODEX_WIKI_INGEST_THRESHOLD", raising=False)
    monkeypatch.setattr(
        settings,
        "_get_section",
        lambda section: {"wiki-ingest-threshold": 0.24},
    )
    assert settings.get_wiki_ingest_threshold() == 0.24


def test_wiki_ingest_threshold_env_takes_priority(monkeypatch) -> None:
    monkeypatch.setenv("IMAS_CODEX_WIKI_INGEST_THRESHOLD", "0.31")
    monkeypatch.setattr(
        settings,
        "_get_section",
        lambda section: {"wiki-ingest-threshold": 0.24},
    )
    assert settings.get_wiki_ingest_threshold() == 0.31
