"""Controlled paired-corpus construction and frozen-artifact invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from imas_codex.standard_names import benchmark_roles

_CORPUS_PATH = Path(__file__).parent / "eval_sets" / "advisory_paired_corpus.json"


def _candidate(name: str, domain: str) -> dict:
    return {
        "name": name,
        "description": f"Physical meaning of {name}.",
        "documentation": f"Documented physical meaning of {name} for review." * 2,
        "unit": "1",
        "kind": "scalar",
        "physics_domain": domain,
        "source_paths": [f"dd:{domain}/{name}"],
    }


def test_two_token_vacuous_corruption_changes_plasma_resistance() -> None:
    assert (
        benchmark_roles._seed_bad_name("plasma_resistance", "vacuous", "")
        == "resistance"
    )


def test_builder_refuses_an_unchanged_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        _candidate("alpha_measurement", "alpha"),
        _candidate("plasma_resistance", "beta"),
        _candidate("unused_foreign_identity", "gamma"),
    ]
    seed_bad_name = benchmark_roles._seed_bad_name

    def leave_plasma_resistance_unchanged(
        good_name: str, defect: str, foreign_name: str
    ) -> str:
        if good_name == "plasma_resistance" and defect == "vacuous":
            return good_name
        return seed_bad_name(good_name, defect, foreign_name)

    monkeypatch.setattr(
        benchmark_roles,
        "_seed_bad_name",
        leave_plasma_resistance_unchanged,
    )

    with pytest.raises(ValueError, match="corruption left identity unchanged"):
        benchmark_roles.build_advisory_paired_corpus(candidates, 2, 20260825)


def test_builder_refuses_duplicate_bad_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        _candidate("alpha_measurement", "alpha"),
        _candidate("beta_measurement", "beta"),
        _candidate("unused_foreign_identity", "gamma"),
    ]
    monkeypatch.setattr(
        benchmark_roles,
        "_seed_bad_name",
        lambda good_name, defect, foreign_name: "repeated_bad_identity",
    )

    with pytest.raises(ValueError, match="corruption produced duplicate identity"):
        benchmark_roles.build_advisory_paired_corpus(candidates, 2, 20260825)


def test_frozen_corpus_has_exact_pairs_provenance_and_hash() -> None:
    payload = json.loads(_CORPUS_PATH.read_text())
    rows = payload["rows"]

    assert payload["seed"] == 20260825
    assert payload["row_count"] == len(rows) == 40
    assert payload["good_count"] == sum(row["label"] == 1 for row in rows) == 20
    assert payload["bad_count"] == sum(row["label"] == 0 for row in rows) == 20
    assert payload["rows_sha256"] == benchmark_roles._canonical_rows_hash(rows)
    assert len({row["standard_name"] for row in rows}) == 40
    extraction = payload["source_extraction"]
    assert extraction["candidate_count"] == 1894
    assert extraction["candidate_domain_count"] == 18
    assert extraction["query"] == benchmark_roles._ADVISORY_CORPUS_QUERY.strip()
    assert (
        extraction["query_sha256"]
        == hashlib.sha256(extraction["query"].encode()).hexdigest()
    )

    pairs: dict[int, list[dict]] = {}
    for row in rows:
        assert row["path"]
        assert row["standard_name"]
        assert row["unit"]
        assert row["provenance"]
        assert row["path"] == row["provenance"]["source_path"]
        pairs.setdefault(row["provenance"]["pair_index"], []).append(row)

    assert set(pairs) == set(range(20))
    for pair in pairs.values():
        assert {row["label"] for row in pair} == {0, 1}
        assert len(pair) == 2
        assert len({row["path"] for row in pair}) == 1
        assert len({row["unit"] for row in pair}) == 1
        assert len({row["provenance"]["candidate_identity"] for row in pair}) == 1
        good = next(row for row in pair if row["label"] == 1)
        bad = next(row for row in pair if row["label"] == 0)
        assert good["standard_name"] != bad["standard_name"]
